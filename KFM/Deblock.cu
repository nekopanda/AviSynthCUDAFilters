
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "Copy.h"
#include "VectorFunctions.cuh"
#include "KFMFilterBase.cuh"

struct QPClipInfo {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x6180FDF8,
  };
  int nMagicKey;
  int nVersion;

  int imageWidth;
  int imageHeight;

  QPClipInfo(const VideoInfo& vi)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , imageWidth(vi.width)
    , imageHeight(vi.height)
  { }

  static const QPClipInfo* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const QPClipInfo* param = (const QPClipInfo*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const QPClipInfo* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

// 映像なしでQPだけのクリップ
class QPClip : public GenericVideoFilter
{
  QPClipInfo info;
public:
  QPClip(PClip clip, IScriptEnvironment* env)
    : GenericVideoFilter(clip)
    , info(vi)
  {
    QPClipInfo::SetParam(vi, &info);

    // フレーム自体はダミー
    vi.width = 2;
    vi.height = 2;
    vi.pixel_type = VideoInfo::CS_Y8;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);
    env->CopyFrameProps(src, dst);
    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new QPClip(
      args[0].AsClip(),      // clip
      env
    );
  }
};

struct CPUInfo {
  bool initialized, avx, avx2;
};

static CPUInfo g_cpuinfo;

static inline void InitCPUInfo() {
  if (g_cpuinfo.initialized == false) {
    int cpuinfo[4];
    __cpuid(cpuinfo, 1);
    g_cpuinfo.avx = cpuinfo[2] & (1 << 28) || false;
    bool osxsaveSupported = cpuinfo[2] & (1 << 27) || false;
    g_cpuinfo.avx2 = false;
    if (osxsaveSupported && g_cpuinfo.avx)
    {
      // _XCR_XFEATURE_ENABLED_MASK = 0
      unsigned long long xcrFeatureMask = _xgetbv(0);
      g_cpuinfo.avx = (xcrFeatureMask & 0x6) == 0x6;
      if (g_cpuinfo.avx) {
        __cpuid(cpuinfo, 7);
        g_cpuinfo.avx2 = cpuinfo[1] & (1 << 5) || false;
      }
    }
    g_cpuinfo.initialized = true;
  }
}

bool IsAVXAvailable() {
  InitCPUInfo();
  return g_cpuinfo.avx;
}

bool IsAVX2Available() {
  InitCPUInfo();
  return g_cpuinfo.avx2;
}

// got from the following command in python. (do "from math import *" before)
#define S1    0.19509032201612825f   // sin(1*pi/(2*8))
#define C1    0.9807852804032304f    // cos(1*pi/(2*8))
#define S3    0.5555702330196022f    // sin(3*pi/(2*8))
#define C3    0.8314696123025452f    // cos(3*pi/(2*8))
#define S2S6  1.3065629648763766f    // sqrt(2)*sin(6*pi/(2*8))
#define S2C6  0.5411961001461971f    // sqrt(2)*cos(6*pi/(2*8))
#define S2    1.4142135623730951f    // sqrt(2)

template <int stride>
__device__ __host__ void dev_dct8(float* data)
{
  // stage 1
  float a0 = data[7 * stride] + data[0 * stride];
  float a1 = data[6 * stride] + data[1 * stride];
  float a2 = data[5 * stride] + data[2 * stride];
  float a3 = data[4 * stride] + data[3 * stride];
  float a4 = data[3 * stride] - data[4 * stride];
  float a5 = data[2 * stride] - data[5 * stride];
  float a6 = data[1 * stride] - data[6 * stride];
  float a7 = data[0 * stride] - data[7 * stride];

  // stage 2 even
  float b0 = a3 + a0;
  float b1 = a2 + a1;
  float b2 = a1 - a2;
  float b3 = a0 - a3;

  // stage 2 odd
  float b4 = (S3 - C3) * a7 + C3 * (a4 + a7);
  float b5 = (S1 - C1) * a6 + C1 * (a5 + a6);
  float b6 = -(C1 + S1) * a5 + C1 * (a5 + a6);
  float b7 = -(C3 + S3) * a4 + C3 * (a4 + a7);

  // stage3 even
  float c0 = b1 + b0;
  float c1 = b0 - b1;
  float c2 = (S2S6 - S2C6) * b3 + S2C6 * (b2 + b3);
  float c3 = -(S2C6 + S2S6) * b2 + S2C6 * (b2 + b3);

  // stage3 odd
  float c4 = b6 + b4;
  float c5 = b7 - b5;
  float c6 = b4 - b6;
  float c7 = b5 + b7;

  // stage 4 odd
  float d4 = c7 - c4;
  float d5 = c5 * S2;
  float d6 = c6 * S2;
  float d7 = c4 + c7;

  // store
  data[0 * stride] = c0;
  data[4 * stride] = c1;
  data[2 * stride] = c2;
  data[6 * stride] = c3;
  data[7 * stride] = d4;
  data[3 * stride] = d5;
  data[5 * stride] = d6;
  data[1 * stride] = d7;
}

template <int stride>
__device__ __host__ void dev_idct8(float* data)
{
  float c0 = data[0 * stride];
  float c1 = data[4 * stride];
  float c2 = data[2 * stride];
  float c3 = data[6 * stride];
  float d4 = data[7 * stride];
  float d5 = data[3 * stride];
  float d6 = data[5 * stride];
  float d7 = data[1 * stride];

  float c4 = d7 - d4;
  float c5 = d5 * S2;
  float c6 = d6 * S2;
  float c7 = d4 + d7;

  float b0 = c1 + c0;
  float b1 = c0 - c1;
  float b2 = -(S2C6 + S2S6) * c3 + S2C6 * (c2 + c3);
  float b3 = (S2S6 - S2C6) * c2 + S2C6 * (c2 + c3);

  float b4 = c6 + c4;
  float b5 = c7 - c5;
  float b6 = c4 - c6;
  float b7 = c5 + c7;

  float a0 = b3 + b0;
  float a1 = b2 + b1;
  float a2 = b1 - b2;
  float a3 = b0 - b3;

  float a4 = -(C3 + S3) * b7 + C3 * (b4 + b7);
  float a5 = -(C1 + S1) * b6 + C1 * (b5 + b6);
  float a6 = (S1 - C1) * b5 + C1 * (b5 + b6);
  float a7 = (S3 - C3) * b4 + C3 * (b4 + b7);

  data[0 * stride] = a7 + a0;
  data[1 * stride] = a6 + a1;
  data[2 * stride] = a5 + a2;
  data[3 * stride] = a4 + a3;
  data[4 * stride] = a3 - a4;
  data[5 * stride] = a2 - a5;
  data[6 * stride] = a1 - a6;
  data[7 * stride] = a0 - a7;
}

__device__ void dev_dct8x8(int tx, float* data)
{
  dev_dct8<1>(data + tx * 9); // row
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
  dev_dct8<9>(data + tx);  // column
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
}

__device__ void dev_idct8x8(int tx, float* data)
{
  dev_idct8<9>(data + tx);  // column
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
  dev_idct8<1>(data + tx * 9); // row
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
}

__host__ void cpu_dct8x8(float* data)
{
  for (int i = 0; i < 8; ++i)
    dev_dct8<1>(data + i * 8); // row
  for (int i = 0; i < 8; ++i)
    dev_dct8<8>(data + i);  // column
}

__host__ void cpu_idct8x8(float* data)
{
  for (int i = 0; i < 8; ++i)
    dev_idct8<8>(data + i);  // column
  for (int i = 0; i < 8; ++i)
    dev_idct8<1>(data + i * 8); // row
}

__device__ void dev_hardthresh(int tx, float *data, float threshold)
{
  for (int i = tx; i < 72; i += 8) {
    if (i == 0) continue;
    float level = data[i];
    if (abs(level) <= threshold) {
      data[i] = 0;
    }
  }
}

//__device__ void dev_softthresh(int tx, float *data, float threshold)
//{
//  for (int i = tx; i < 72; i += 8) {
//    if (i == 0) continue;
//    float level = data[i];
//    if (abs(level) <= threshold) data[i] = 0;
//    else if (level > 0) data[i] -= threshold;
//    else                data[i] += threshold;
//  }
//}

__host__ void cpu_hardthresh(float *data, float threshold)
{
  for (int i = 1; i < 64; ++i) {
    float level = data[i];
    if (abs(level) <= threshold) {
      data[i] = 0;
    }
  }
}

//__host__ void cpu_softthresh(float *data, float threshold)
//{
//  for (int i = 1; i < 64; ++i) {
//    float level = data[i];
//    if (abs(level) <= threshold) data[i] = 0;
//    else if (level > 0) data[i] -= threshold;
//    else                data[i] += threshold;
//  }
//}

__device__ __host__ float qp_apply_thresh(float qp, float thresh_a, float thresh_b)
{
  return clamp(thresh_a * qp + thresh_b, 0.0f, qp);
}

#ifdef __CUDA_ARCH__
__constant__
#endif
uchar2 g_deblock_offset[127] = {
  { 0,0 },
  { 0,0 },{ 4,4 },                                           // quality = 1
  { 0,0 },{ 2,2 },{ 6,4 },{ 4,6 },                             // quality = 2
  { 0,0 },{ 5,1 },{ 2,2 },{ 7,3 },{ 4,4 },{ 1,5 },{ 6,6 },{ 3,7 }, // quality = 3

  { 0,0 },{ 4,0 },{ 1,1 },{ 5,1 },{ 3,2 },{ 7,2 },{ 2,3 },{ 6,3 }, // quality = 4
  { 0,4 },{ 4,4 },{ 1,5 },{ 5,5 },{ 3,6 },{ 7,6 },{ 2,7 },{ 6,7 },

  { 0,0 },{ 0,2 },{ 0,4 },{ 0,6 },{ 1,1 },{ 1,3 },{ 1,5 },{ 1,7 }, // quality = 5
  { 2,0 },{ 2,2 },{ 2,4 },{ 2,6 },{ 3,1 },{ 3,3 },{ 3,5 },{ 3,7 },
  { 4,0 },{ 4,2 },{ 4,4 },{ 4,6 },{ 5,1 },{ 5,3 },{ 5,5 },{ 5,7 },
  { 6,0 },{ 6,2 },{ 6,4 },{ 6,6 },{ 7,1 },{ 7,3 },{ 7,5 },{ 7,7 },

  { 0,0 },{ 4,4 },{ 0,4 },{ 4,0 },{ 2,2 },{ 6,6 },{ 2,6 },{ 6,2 }, // quality = 6
  { 0,2 },{ 4,6 },{ 0,6 },{ 4,2 },{ 2,0 },{ 6,4 },{ 2,4 },{ 6,0 },
  { 1,1 },{ 5,5 },{ 1,5 },{ 5,1 },{ 3,3 },{ 7,7 },{ 3,7 },{ 7,3 },
  { 1,3 },{ 5,7 },{ 1,7 },{ 5,3 },{ 3,1 },{ 7,5 },{ 3,5 },{ 7,1 },
  { 0,1 },{ 4,5 },{ 0,5 },{ 4,1 },{ 2,3 },{ 6,7 },{ 2,7 },{ 6,3 },
  { 0,3 },{ 4,7 },{ 0,7 },{ 4,3 },{ 2,1 },{ 6,5 },{ 2,5 },{ 6,1 },
  { 1,0 },{ 5,4 },{ 1,4 },{ 5,0 },{ 3,2 },{ 7,6 },{ 3,6 },{ 7,2 },
  { 1,2 },{ 5,6 },{ 1,6 },{ 5,2 },{ 3,0 },{ 7,4 },{ 3,4 },{ 7,0 },
};

// src: 外周8ピクセル拡張したソース
// out: 外周8ピクセル拡張し、かつ、縦方向に4倍した中間出力バッファ
// qp_table: ffmpegから取得したqpテーブル
// offsets: ブロックオフセットテーブル
// shift: min(3, 16 - quality - bits)
// maxv: (1 << (11 - shift)) - 1
template <typename pixel_t>
__global__ void kl_deblock(
  const pixel_t* __restrict__ src, int src_pitch,
  int bw, int bh,
  ushort2* out, int out_pitch,
  const uint16_t* __restrict__ qp_table, int qp_pitch,
  int count_minus_1, int shift, int maxv,
  float strength, float thresh_a, float thresh_b)
{
  int tx = threadIdx.x; // 8
  int ty = threadIdx.y; // count

  __shared__ ushort2 local_out[16][8];
  extern __shared__ float dct_tmp_buf[];

  // sharedメモリはバンクコンフリクトを回避するため9x8で確保する
  float* dct_tmp = dct_tmp_buf + 72 * ty;

  // local_out初期化
  for (int y = ty; y < 16; y += blockDim.y) {
    local_out[y][tx] = ushort2();
  }
  __syncthreads();

  // getpixel
  const uchar2 offset = g_deblock_offset[count_minus_1 + ty];
  {
    int off_x = blockIdx.x * 8 + offset.x;
    int off_y = blockIdx.y * 8 + offset.y;
    for (int y = 0; y < 8; ++y) {
      dct_tmp[tx + y * 9] = src[(off_x + tx) + (off_y + y) * src_pitch];
    }
  }

  // dct
  dev_dct8x8(tx, dct_tmp);

  // requantize
  uint16_t qp = qp_table[blockIdx.x + blockIdx.y * qp_pitch];
  float thresh = qp_apply_thresh(qp, thresh_a, thresh_b) * ((1 << 2) + strength) - 1;
  // dev_softthresh(tx, dct_tmp, thresh);
  dev_hardthresh(tx, dct_tmp, thresh);
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif

  // idct
  dev_idct8x8(tx, dct_tmp);

  // add
  // 16bitのatomicAddはないので、2つの16bitを1つの32bitとして扱う
  //（オーバーフローしない前提）
  const int half = (1 << shift) >> 1;
  for (int y = 0; y < 8; ++y) {
    int tmp = clamp((int)(dct_tmp[tx + y * 9] + half) >> shift, 0, maxv);
    int off_x = offset.x + tx;
    atomicAdd((int32_t*)&local_out[offset.y + y][off_x >> 1], tmp << ((off_x & 1) * 16));
  }
  __syncthreads();

  // store
  int off_z = (blockIdx.x & 1) + (blockIdx.y & 1) * 2;
  int off_x = blockIdx.x * 4;
  int off_y = (bh * off_z + blockIdx.y) * 8;
  for (int y = ty; y < 16; y += blockDim.y) {
    out[(off_x + tx) + (off_y + y) * out_pitch] = local_out[y][tx];
  }
}

template <typename pixel_t>
void cpu_deblock(
  const pixel_t* src, int src_pitch,
  int bw, int bh,
  uint16_t* out, int out_pitch,
  const uint16_t* qp_table, int qp_pitch,
  int count_minus_1, int shift, int maxv,
  float strength, float thresh_a, float thresh_b)
{
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      uint16_t local_out[16][16];

      // local_out初期化
      memset(local_out, 0, sizeof(local_out));

      for (int ty = 0; ty <= count_minus_1; ++ty) {
        // getpixel
        const uchar2 offset = g_deblock_offset[count_minus_1 + ty];
        int off_x = bx * 8 + offset.x;
        int off_y = by * 8 + offset.y;
        float dct_tmp[64];
        for (int y = 0; y < 8; ++y) {
          for (int x = 0; x < 8; ++x) {
            dct_tmp[x + y * 8] = src[(off_x + x) + (off_y + y) * src_pitch];
          }
        }

        uint16_t qp = qp_table[bx + by * qp_pitch];
        float thresh = qp_apply_thresh(qp, thresh_a, thresh_b) * ((1 << 2) + strength) - 1;

        if (thresh <= 0) {
          // 変化しないのでそのまま入れる
          // といってもdct->idctで8*8倍されるので64倍
          for (int i = 0; i < 64; ++i) {
            dct_tmp[i] *= 64;
          }
        }
        else {
          // dct
          cpu_dct8x8(dct_tmp);

          // requantize
          // cpu_softthresh(dct_tmp, thresh);
          cpu_hardthresh(dct_tmp, thresh);

          // idct
          cpu_idct8x8(dct_tmp);
        }

        // add
        const int half = (1 << shift) >> 1;
        for (int y = 0; y < 8; ++y) {
          for (int x = 0; x < 8; ++x) {
            auto tmp = clamp((int)(dct_tmp[x + y * 8] + half) >> shift, 0, maxv);
            local_out[offset.y + y][offset.x + x] += tmp;
#if 0
            if ((bx - 1) * 8 + offset.x + x == 723 && (by - 1) * 8 + offset.y + y == 590) {
              printf("REF: %d (%d)\n", tmp, local_out[offset.y + y][offset.x + x]);
            }
#endif
          }
        }
      }

      // store
      int off_z = (bx & 1) + (by & 1) * 2;
      int off_x = bx * 8;
      int off_y = (bh * off_z + by) * 8;
      for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
          out[(off_x + x) + (off_y + y) * out_pitch] = local_out[y][x];
        }
      }
    }
  }
}

template <typename pixel_t>
__global__ void kl_deblock_show(
  pixel_t* dst, int dst_pitch,
  int width, int height,
  int bw, int bh,
  const uint16_t* qp_table, int qp_pitch,
  float thresh_a, float thresh_b)
{
  uint16_t qp = qp_table[blockIdx.x + blockIdx.y * qp_pitch];
  bool is_enabled = (qp_apply_thresh(qp, thresh_a, thresh_b) >= (qp >> 1));
  int off_x = blockIdx.x * 8 - 4;
  int off_y = blockIdx.y * 8 - 4;
  int x = off_x + threadIdx.x;
  int y = off_y + threadIdx.y;
  if (x >= 0 && x < width && y >= 0 && y < height) {
    dst[x + y * dst_pitch] = is_enabled ? 230 : 16;
  }
}

template <typename pixel_t>
void cpu_deblock_show(
  pixel_t* dst, int dst_pitch,
  int width, int height,
  int bw, int bh,
  const uint16_t* qp_table, int qp_pitch,
  float thresh_a, float thresh_b)
{
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      uint16_t qp = qp_table[bx + by * qp_pitch];
      bool is_enabled = (qp_apply_thresh(qp, thresh_a, thresh_b) >= (qp >> 1));
      int off_x = bx * 8 - 4;
      int off_y = by * 8 - 4;
      for (int dy = 0; dy < 8; ++dy) {
        for (int dx = 0; dx < 8; ++dx) {
          int x = off_x + dx;
          int y = off_y + dy;
          if (x >= 0 && x < width && y >= 0 && y < height) {
            dst[x + y * dst_pitch] = is_enabled ? 230 : 16;
          }
        }
      }
    }
  }
}

// normalize the qscale factor
// ffmpegはmpeg1に合わせているが値が小さいのでh264に合わせるようにした
//（ffmpegの4倍の値が返る）
__device__ __host__ inline int norm_qscale(int qscale, int type)
{
  switch (type) {
  case 0/*FF_QSCALE_TYPE_MPEG1*/: return qscale << 2;
  case 1/*FF_QSCALE_TYPE_MPEG2*/: return qscale << 1;
  case 2/*FF_QSCALE_TYPE_H264*/:  return qscale;
  case 3/*FF_QSCALE_TYPE_VP56*/:  return (63 - qscale + 2);
  }
  return qscale;
}


__global__ void kl_make_qp_table(
  int in_width, int in_height,
  const uint8_t* in_table0, const uint8_t* nonb_table0,
	const uint8_t* in_table1, const uint8_t* nonb_table1,
  int in_pitch, int qp_scale, float dc_coeff,
	const uint8_t* dc_table0, const uint8_t* dc_table1, int dc_pitch,
  int qp_shift_x, int qp_shift_y,
  int out_width, int out_height,
  uint16_t* out_table, int out_pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < out_width && y < out_height) {
    int qp;
		if (in_table0) {
			int qp_x = min(x >> qp_shift_x, in_width - 1);
			int qp_y = min(y >> qp_shift_y, in_height - 1);
			int in_qp = in_table0[qp_x + qp_y * in_pitch];
			int nonb_qp = nonb_table0[qp_x + qp_y * in_pitch];
			int dc = dc_table0 ? dc_table0[qp_x + qp_y * dc_pitch] : 255;
			if (in_table1) {
				in_qp = max(in_qp, (int)in_table1[qp_x + qp_y * in_pitch]);
				nonb_qp = max(nonb_qp, (int)nonb_table1[qp_x + qp_y * in_pitch]);
				dc = max(dc, dc_table1 ? dc_table1[qp_x + qp_y * dc_pitch] : 255);
			}
			int b = norm_qscale(in_qp, qp_scale);
			int nonb = norm_qscale(nonb_qp, qp_scale);
			float b_ratio = min(1.0f, dc * dc_coeff);
			qp = max(1, (int)(b * b_ratio + nonb * (1 - b_ratio) + 0.5f));
		}
    else {
      qp = qp_scale;
    }
    out_table[x + y * out_pitch] = qp;
  }
}

void cpu_make_qp_table(
  int in_width, int in_height,
	const uint8_t* in_table0, const uint8_t* nonb_table0,
	const uint8_t* in_table1, const uint8_t* nonb_table1,
  int in_pitch, int qp_scale, float dc_coeff,
	const uint8_t* dc_table0, const uint8_t* dc_table1, int dc_pitch,
  int qp_shift_x, int qp_shift_y,
  int out_width, int out_height,
  uint16_t* out_table, int out_pitch)
{
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      int qp;
      if (in_table0) {
        int qp_x = min(x >> qp_shift_x, in_width - 1);
        int qp_y = min(y >> qp_shift_y, in_height - 1);
        int in_qp = in_table0[qp_x + qp_y * in_pitch];
        int nonb_qp = nonb_table0[qp_x + qp_y * in_pitch];
				int dc = dc_table0 ? dc_table0[qp_x + qp_y * dc_pitch] : 255;
        if (in_table1) {
          in_qp = max(in_qp, (int)in_table1[qp_x + qp_y * in_pitch]);
          nonb_qp = max(nonb_qp, (int)nonb_table1[qp_x + qp_y * in_pitch]);
					dc = max(dc, dc_table1 ? dc_table1[qp_x + qp_y * dc_pitch] : 255);
        }
        int b = norm_qscale(in_qp, qp_scale);
        int nonb = norm_qscale(nonb_qp, qp_scale);
				float b_ratio = min(1.0f, dc * dc_coeff);
        qp = max(1, (int)(b * b_ratio + nonb * (1 - b_ratio) + 0.5f));
      }
      else {
        qp = qp_scale;
      }
      out_table[x + y * out_pitch] = qp;
    }
  }
}

#ifdef __CUDA_ARCH__
__constant__
#endif
uchar4 g_ldither[8][2] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

// shift: quality - (3 - deblock_shift)
template <typename vpixel_t>
__global__ void kl_merge_deblock(
  int width, int height,
  const ushort4 *tmp, int tmp_pitch, int tmp_ipitch,
  vpixel_t* out, int out_pitch, int shift, float maxv)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto sum = to_int(tmp[x + (tmp_ipitch * 0 + y) * tmp_pitch]) +
      to_int(tmp[x + (tmp_ipitch * 1 + y) * tmp_pitch]) +
      to_int(tmp[x + (tmp_ipitch * 2 + y) * tmp_pitch]) +
      to_int(tmp[x + (tmp_ipitch * 3 + y) * tmp_pitch]);
    auto tmp = to_float(sum) * (1.0f / (1 << shift)) +
      to_float(g_ldither[y & 7][x & 1]) * (1.0f / 64.0f);
    out[x + y * out_pitch] = VHelper<vpixel_t>::cast_to(min(tmp, maxv));
  }
}

template <typename vpixel_t>
void cpu_merge_deblock(
  int width, int height,
  const ushort4 *tmp, int tmp_pitch, int tmp_ipitch,
  vpixel_t* out, int out_pitch, int shift, float maxv)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto sum = to_int(tmp[x + (tmp_ipitch * 0 + y) * tmp_pitch]) +
        to_int(tmp[x + (tmp_ipitch * 1 + y) * tmp_pitch]) +
        to_int(tmp[x + (tmp_ipitch * 2 + y) * tmp_pitch]) +
        to_int(tmp[x + (tmp_ipitch * 3 + y) * tmp_pitch]);
      auto tmp = to_float(sum) * (1.0f / (1 << shift)) +
        to_float(g_ldither[y & 7][x & 1]) * (1.0f / 64.0f);
      out[x + y * out_pitch] = VHelper<vpixel_t>::cast_to(min(tmp, maxv));
    }
  }
}

template <typename pixel_t>
void cpu_deblock_kernel_avx(const pixel_t* src, int src_pitch,
  uint16_t* dst, int dst_pitch, float thresh, float half, int shift, int maxv);

template <typename pixel_t>
void cpu_store_slice_avx(
  int width, int height, pixel_t* dst, int dst_pitch,
  const uint16_t* tmp, int tmp_pitch, int shift, int maxv);

template <typename pixel_t>
void cpu_deblock_avx(
  const pixel_t* src, int src_pitch,
  int bw, int bh,
  uint16_t* tmp, int tmp_pitch,
  const uint16_t* qp_table, int qp_pitch,
  int count_minus_1, int deblockShift, int deblockMaxV,
  float strength, float thresh_a, float thresh_b,

  int width, int height,
  pixel_t* dst, int dst_pitch, int mergeShift, int mergeMaxV)
{
  for (int by = 0; by < bh; ++by) {
    memset(&tmp[(by + 1) * 8 * tmp_pitch], 0, tmp_pitch * 8 * sizeof(uint16_t)); // dst初期化
    for (int bx = 0; bx < bw; ++bx) {
      for (int ty = 0; ty <= count_minus_1; ++ty) {
        const uchar2 offset = g_deblock_offset[count_minus_1 + ty];
        int off_x = bx * 8 + offset.x;
        int off_y = by * 8 + offset.y;
        uint16_t qp = qp_table[bx + by * qp_pitch];
        float thresh = qp_apply_thresh(qp, thresh_a, thresh_b) * ((1 << 2) + strength) - 1;
        const int half = (1 << deblockShift) >> 1;
        const pixel_t* src_block = &src[off_x + off_y * src_pitch];
        uint16_t* tmp_block = &tmp[off_x + off_y * tmp_pitch];

        cpu_deblock_kernel_avx(src_block, src_pitch,
          tmp_block, tmp_pitch, thresh, (float)half, deblockShift, deblockMaxV);
#if 0
        int x_start = (bx - 1) * 8 + offset.x;
        int y_start = (by - 1) * 8 + offset.y;
        if (x_start <= 723 && x_start + 8 > 723 && y_start <= 590 && y_start + 8 > 590) {
          printf("AVX: ? (%d)\n", tmp[(723 + 8) + (590 + 8) * tmp_pitch]);
        }
#endif
      }
    }
    if (by) {
      int y_start = (by - 1) * 8;
      cpu_store_slice_avx(
        width, min(8, height - y_start), &dst[y_start * dst_pitch], dst_pitch,
        &tmp[by * 8 * tmp_pitch + 8], tmp_pitch, mergeShift, mergeMaxV);
    }
  }
}

template <int RADIUS>
__global__ void kl_max_v(uchar4* dst, uchar4* src, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		uchar4 sum = { 0 };
		for (int i = -RADIUS; i <= RADIUS; ++i) {
			sum = max(sum, src[x + (y + i) * pitch]);
		}
		dst[x + y * pitch] = sum;
	}
}

template <int RADIUS>
void cpu_max_v(uchar4* dst, uchar4* src, int width, int height, int pitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			uchar4 sum = { 0 };
			for (int i = -RADIUS; i <= RADIUS; ++i) {
				sum = max(sum, src[x + (y + i) * pitch]);
			}
			dst[x + y * pitch] = sum;
		}
	}
}

template <int RADIUS>
__global__ void kl_max_h(uint8_t* dst, uint8_t* src, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		uint8_t sum = { 0 };
		for (int i = -RADIUS; i <= RADIUS; ++i) {
			sum = max(sum, src[(x + i) + y * pitch]);
		}
		dst[x + y * pitch] = sum;
	}
}

template <int RADIUS>
void cpu_max_h(uint8_t* dst, uint8_t* src, int width, int height, int pitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			uint8_t sum = { 0 };
			for (int i = -RADIUS; i <= RADIUS; ++i) {
				sum = max(sum, src[(x + i) + y * pitch]);
			}
			dst[x + y * pitch] = sum;
		}
	}
}

bool IsAVX2Available();

enum QP_RESULT_FLAG {
	QP_TABLE_NONE,
	QP_TABLE_ONLY,
	QP_TABLE_CONSTANT,
	QP_TABLE_USING_DC,
};

class QPForDeblock : public KFMFilterBase
{
	VideoInfo srcvi;
	PClip qpclip;
	float frameRateConv; // (qpclip frame rate) / (source frame rate)

	float b_ratio;
	int force_qp;
	bool b_adap;

	void QPForPlane(
		int width, int height,
		uint16_t* dst, int dstPitch,
		const uint8_t* qpTable0, const uint8_t* qpTableNonB0, Frame dc0,
		const uint8_t* qpTable1, const uint8_t* qpTableNonB1, Frame dc1,
		int qpStride, int qpScaleType,
		int qpShiftX, int qpShiftY, PNeoEnv env)
	{
		float dc_coeff = b_ratio / 255.0f;
		int qp_width = (width + 7 + 8) >> 3;
		int qp_height = (height + 7 + 8) >> 3;

		if (IS_CUDA) {
			dim3 threads(32, 8);
			dim3 blocks(nblocks(qp_width, threads.x), nblocks(qp_height, threads.y));
			kl_make_qp_table << <blocks, threads >> > (
				(srcvi.width + 15) >> 4, (srcvi.height + 15) >> 4,
				qpTable0, qpTableNonB0, qpTable1, qpTableNonB1,
				qpStride, qpTable0 ? qpScaleType : force_qp, dc_coeff,
				dc0.GetReadPtr<uint8_t>(), dc1.GetReadPtr<uint8_t>(), dc0.GetPitch<uint8_t>(),
				qpShiftX, qpShiftY, qp_width, qp_height,
				dst, dstPitch);
			DEBUG_SYNC;
		}
		else {
			cpu_make_qp_table((srcvi.width + 15) >> 4, (srcvi.height + 15) >> 4,
				qpTable0, qpTableNonB0, qpTable1, qpTableNonB1,
				qpStride, qpTable0 ? qpScaleType : force_qp, dc_coeff,
				dc0.GetReadPtr<uint8_t>(), dc1.GetReadPtr<uint8_t>(), dc0.GetPitch<uint8_t>(),
				qpShiftX, qpShiftY, qp_width, qp_height,
				dst, dstPitch);
		}
	}

	Frame MakeMask(PVideoFrame dcframe, PNeoEnv env)
	{
		typedef typename VectorType<uint8_t>::type vpixel_t;

		int width = dcframe->GetRowSize();
		int height = dcframe->GetHeight();

		VideoInfo padvi = vi;
		padvi.width = width + 8 * 2;
		padvi.height = height + 8 * 2;
		padvi.pixel_type = VideoInfo::CS_Y8;
		Frame pad = env->NewVideoFrame(padvi);
		Frame tmp = env->NewVideoFrame(padvi);

		Copy(pad.GetWritePtr<uint8_t>() + 8 + 8 * pad.GetPitch<uint8_t>(),
			pad.GetPitch<uint8_t>(), dcframe->GetReadPtr(), dcframe->GetPitch(),
			width, height, env);

		int width4 = (width + 3) >> 2;

		if (IS_CUDA) {
			kl_padv<vpixel_t> << <dim3(nblocks(width4, 32)), dim3(32, 8) >> > (
				pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
				width4, height, pad.GetPitch<vpixel_t>(), 8);
			DEBUG_SYNC;
			kl_padh<uint8_t> << <dim3(1, nblocks(height + 8 * 2, 32)), dim3(8, 32) >> > (
				pad.GetWritePtr<uint8_t>() + 8, width, height + 8 * 2,
				pad.GetPitch<uint8_t>(), 8);
			DEBUG_SYNC;
		}
		else {
			cpu_padv<vpixel_t>(
				pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
				width >> 2, height, pad.GetPitch<vpixel_t>(), 8);
			cpu_padh<uint8_t>(
				pad.GetWritePtr<uint8_t>() + 8, width, height + 8 * 2,
				pad.GetPitch<uint8_t>(), 8);
		}

		if (IS_CUDA) {
			kl_max_v<5> << <dim3(nblocks(width4, 32), nblocks(height, 8)), dim3(32, 8) >> > (
				tmp.GetWritePtr<vpixel_t>() + 2 + 8 * tmp.GetPitch<vpixel_t>(),
				pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
				width4, height, pad.GetPitch<vpixel_t>());
			DEBUG_SYNC;
			kl_max_h<5> << <dim3(nblocks(width, 32), nblocks(height, 8)), dim3(32, 8) >> > (
				pad.GetWritePtr<uint8_t>() + 8 + 8 * pad.GetPitch<uint8_t>(),
				tmp.GetWritePtr<uint8_t>() + 8 + 8 * tmp.GetPitch<uint8_t>(),
				width, height, pad.GetPitch<uint8_t>());
			DEBUG_SYNC;
		}
		else {
			cpu_max_v<5>(
				tmp.GetWritePtr<vpixel_t>() + 2 + 8 * tmp.GetPitch<vpixel_t>(),
				pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
				width4, height, pad.GetPitch<vpixel_t>());
			cpu_max_h<5>(
				pad.GetWritePtr<uint8_t>() + 8 + 8 * pad.GetPitch<uint8_t>(),
				tmp.GetWritePtr<uint8_t>() + 8 + 8 * tmp.GetPitch<uint8_t>(),
				width, height, pad.GetPitch<uint8_t>());
		}

		return Frame(pad.frame, 8, 8, 1);
	}

	PVideoFrame QPEntry(
		const uint8_t* qpTable0, const uint8_t* qpTableNonB0, PVideoFrame dc0,
		const uint8_t* qpTable1, const uint8_t* qpTableNonB1, PVideoFrame dc1,
		int qpStride, int qpScaleType, PNeoEnv env)
	{
		Frame dst = env->NewVideoFrame(vi);
		Frame bmask0 = dc0 ? MakeMask(dc0, env) : Frame();
		Frame bmask1 = dc1 ? MakeMask(dc1, env) : Frame();

		QPForPlane(srcvi.width, srcvi.height,
			dst.GetWritePtr<uint16_t>(PLANAR_Y), dst.GetPitch<uint16_t>(PLANAR_Y),
			qpTable0, qpTableNonB0, bmask0, qpTable1, qpTableNonB1, bmask1,
			qpStride, qpScaleType, 1, 1, env);

		QPForPlane(srcvi.width >> logUVx, srcvi.height >> logUVy,
			dst.GetWritePtr<uint16_t>(PLANAR_U), dst.GetPitch<uint16_t>(PLANAR_U),
			qpTable0, qpTableNonB0, bmask0, qpTable1, qpTableNonB1, bmask1,
			qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, env);

		QPForPlane(srcvi.width >> logUVx, srcvi.height >> logUVy,
			dst.GetWritePtr<uint16_t>(PLANAR_V), dst.GetPitch<uint16_t>(PLANAR_V),
			qpTable0, qpTableNonB0, bmask0, qpTable1, qpTableNonB1, bmask1,
			qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, env);

		dst.SetProperty("DEBLOCK_QP_FLAG", 
			(int)(bmask0 ? QP_TABLE_USING_DC : qpTable0 ? QP_TABLE_ONLY : QP_TABLE_CONSTANT));

		return dst.frame;
	}

	// QPテーブルのフレーム指定
	PVideoFrame QPEntry(PVideoFrame& qp0, PVideoFrame& qp1, PNeoEnv env)
	{
		if (qp0->GetProperty("QP_Table_Non_B") == nullptr) {
			// QPテーブルがない
			Frame dst = env->NewVideoFrame(vi);
			dst.SetProperty("DEBLOCK_QP_FLAG", QP_TABLE_NONE);
			return dst.frame;
		}

		auto OptGetFrame = [](const AVSMapValue* value) {
			return value ? value->GetFrame() : nullptr;
		};

		Frame qpTable0 = qp0->GetProperty("QP_Table")->GetFrame();
		Frame qpTableNonB0 = qp0->GetProperty("QP_Table_Non_B")->GetFrame();
		int qpStride = (int)qp0->GetProperty("QP_Stride")->GetInt();
		int qpScaleType = (int)qp0->GetProperty("QP_ScaleType")->GetInt();
		const AVSMapValue* dc0 = b_adap ? qp0->GetProperty("DC_Table") : nullptr;

		if (!qp1) {
			return QPEntry(
				qpTable0.GetReadPtr<uint8_t>(), qpTableNonB0.GetReadPtr<uint8_t>(), OptGetFrame(dc0),
				nullptr, nullptr, nullptr,
				qpStride, qpScaleType, env);
		}

		Frame qpTable1 = qp1->GetProperty("QP_Table")->GetFrame();
		Frame qpTableNonB1 = qp1->GetProperty("QP_Table_Non_B")->GetFrame();
		const AVSMapValue* dc1 = b_adap ? qp1->GetProperty("DC_Table") : nullptr;

		return QPEntry(
			qpTable0.GetReadPtr<uint8_t>(), qpTableNonB0.GetReadPtr<uint8_t>(), OptGetFrame(dc0),
			qpTable1.GetReadPtr<uint8_t>(), qpTableNonB1.GetReadPtr<uint8_t>(), OptGetFrame(dc1),
			qpStride, qpScaleType, env);
	}

	PVideoFrame GetQPFrame(int n, PNeoEnv env)
	{
		if (qpclip) {
			return qpclip->GetFrame(n, env);
		}
		return child->GetFrame(n, env);
	}

public:
	QPForDeblock(PClip source, float b_ratio,
		PClip qpclip, int force_qp, bool b_adap, IScriptEnvironment* env)
		: KFMFilterBase(source)
		, qpclip(qpclip)
		, b_ratio(b_ratio)
		, force_qp(force_qp)
		, b_adap(b_adap)
	{
		if (vi.width & 7) env->ThrowError("[KDeblock]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KDeblock]: height must be multiple of 8");

		srcvi = vi;

		if (qpclip) {
			auto qpvi = qpclip->GetVideoInfo();
			frameRateConv = (float)(qpvi.fps_numerator * vi.fps_denominator) /
				(float)(qpvi.fps_denominator * vi.fps_numerator);
		}
		else {
			frameRateConv = 1.0f;
		}

		auto calc_qp_size = [](int len) {
			return std::max((len + 7 + 8) >> 3, (((len >> 1) + 7 + 8) >> 3) * 2);
		};

		vi.width = calc_qp_size(vi.width);
		vi.height = calc_qp_size(vi.height);
		vi.pixel_type = Get16BitType(vi);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		PNeoEnv env = env_;
		Frame src = child->GetFrame(n, env);

		if (force_qp > 0) {
			// QP指定あり
			return QPEntry(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, env);
		}

		if (src.GetProperty("KFM_SourceStart")) {
			// フレーム指定あり
			int start = (int)src.GetProperty("KFM_SourceStart")->GetInt();
			int num = (int)src.GetProperty("KFM_NumSourceFrames")->GetInt();

			// 想定されるフレームと離れすぎてたらバグの可能性が高いので検出
			int qp_n = (int)(frameRateConv * n + 0.3f);
			if (start < qp_n - 10 || start > qp_n + 10) {
				env->ThrowError("Invalid KFM_SourceStart");
			}

			PVideoFrame dst;
			if (num <= 1) {
				dst = QPEntry(GetQPFrame(start, env), PVideoFrame(), env);
			}
			else {
				dst = QPEntry(GetQPFrame(start, env), GetQPFrame(start + 1, env), env);
			}
			return dst;
		}

		if (qpclip) {
			// QPクリップ指定あり
			int qp_n = (int)(frameRateConv * n + 0.3f);
			return QPEntry(GetQPFrame(qp_n, env), PVideoFrame(), env);
		}

		// ソースフレームからQP取得
		return QPEntry(src.frame, PVideoFrame(), env);
	}

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_MTMODE) {
			return MT_NICE_FILTER;
		}
		return KFMFilterBase::SetCacheHints(cachehints, frame_range);
	}
};

__constant__ uint8_t d_sharpen_coeff[] = {
	  0,   0,   0,   0,   0, // 0
	  0,   0,   0,   0,  10, // 5(40)
	 50,  90, 120, 150, 160, // 10(80)
	170, 180, 190, 200, 210, // 15(120)
	220, 230, 240, 245, 250, // 20(160)
	255, 255, 255, 255, 255, // 25(200)
};

uint8_t g_sharpen_coeff[] = {
	  0,   0,   0,   0,   0, // 0
	  0,   0,   0,   0,  10, // 5(40)
	 50,  90, 120, 150, 160, // 10(80)
	170, 180, 190, 200, 210, // 15(120)
	220, 230, 240, 245, 250, // 20(160)
	255, 255, 255, 255, 255, // 25(200)
};

__global__ void kl_sharpen_coeff(uint8_t* dst, 
	int width, int height, int pitch, const uint16_t* qp, int qpPitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int q = qp[x + y * qpPitch] >> 3;
		dst[x + y * pitch] = (q >= 25) ? 255 : d_sharpen_coeff[q];
	}
}

void cpu_sharpen_coeff(uint8_t* dst,
	int width, int height, int pitch, const uint16_t* qp, int qpPitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int q = qp[x + y * qpPitch] >> 3;
			dst[x + y * pitch] = (q >= 25) ? 255 : g_sharpen_coeff[q];
		}
	}
}

template <typename pixel_t>
__global__ void kl_sharpen(pixel_t* dst, int width, int height, int pitch,
	cudaTextureObject_t src, cudaTextureObject_t coeff, const pixel_t* unsharp)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int s = tex2D<pixel_t>(src, x, y);
		int l = s;
		int h = s;
		int v;

		v = tex2D<pixel_t>(src, x - 1, y - 1);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x + 0, y - 1);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x + 1, y - 1);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x - 1, y + 0);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x + 1, y + 0);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x - 1, y + 1);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x + 0, y + 1);
		l = min(l, v); h = max(h, v);
		v = tex2D<pixel_t>(src, x + 1, y + 1);
		l = min(l, v); h = max(h, v);

		float c = tex2D<float>(coeff, x * (1.0f / 8.0f) + 0.5f, y * (1.0f / 8.0f) + 0.5f);
		int u = unsharp[x + y * pitch];
		dst[x + y * pitch] = (int)clamp<float>(s + (s - u) * c + 0.5f, l, h);
	}
}

template <typename pixel_t>
void cpu_sharpen(pixel_t* dst, int width, int height, int pitch,
	const pixel_t* src, const uint8_t* coeff, int coeffPitch, const pixel_t* unsharp)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int s = src[x + y * pitch];
			float c;

			{
				float fx = x * (1.0f / 8.0f);
				float fy = y * (1.0f / 8.0f);
				int ix = (int)fx;
				int iy = (int)fy;
				auto c00 = coeff[ix + iy * coeffPitch];
				auto c01 = coeff[(ix + 1) + iy * coeffPitch];
				auto c10 = coeff[ix + (iy + 1) * coeffPitch];
				auto c11 = coeff[(ix + 1) + (iy + 1) * coeffPitch];
				float fracx = fx - ix;
				float fracy = fy - iy;
				c = (c00 * (1 - fracx) + c01 * fracx) * (1 - fracy) + (c10 * (1 - fracx) + c11 * fracx) * fracy;
				c *= 1.0f / 255.0f;
			}

			if (c > 0) {
				int l = s;
				int h = s;
				int v;

				if (y > 0) {
					if (x > 0) {
						v = src[(x - 1) + (y - 1) * pitch];
						l = min(l, v); h = max(h, v);
					}
					v = src[(x + 0) + (y - 1) * pitch];
					l = min(l, v); h = max(h, v);
					if (x < width - 1) {
						v = src[(x + 1) + (y - 1) * pitch];
						l = min(l, v); h = max(h, v);
					}
				}
				if (x > 0) {
					v = src[(x - 1) + (y + 0) * pitch];
					l = min(l, v); h = max(h, v);
				}
				if (x < width - 1) {
					v = src[(x + 1) + (y + 0) * pitch];
					l = min(l, v); h = max(h, v);
				}
				if (y < height - 1) {
					if (x > 0) {
						v = src[(x - 1) + (y + 1) * pitch];
						l = min(l, v); h = max(h, v);
					}
					v = src[(x + 0) + (y + 1) * pitch];
					l = min(l, v); h = max(h, v);
					if (x < width - 1) {
						v = src[(x + 1) + (y + 1) * pitch];
						l = min(l, v); h = max(h, v);
					}
				}

				int u = unsharp[x + y * pitch];
				dst[x + y * pitch] = (int)clamp<float>(s + (s - u) * c + 0.5f, (float)l, (float)h);
			}
			else {
				dst[x + y * pitch] = s;
			}
		}
	}
}

template <typename pixel_t>
__global__ void kl_show_sharpen_coeff(pixel_t* dst, int width, int height, int pitch,
	cudaTextureObject_t coeff)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		float c = tex2D<float>(coeff, x * (1.0f / 8.0f) + 0.5f, y * (1.0f / 8.0f) + 0.5f);
		dst[x + y * pitch] = (int)(c * 255);
	}
}

template <typename pixel_t>
void cpu_show_sharpen_coeff(pixel_t* dst, int width, int height, int pitch,
	const uint8_t* coeff, int coeffPitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float fx = x * (1.0f / 8.0f);
			float fy = y * (1.0f / 8.0f);
			int ix = (int)fx;
			int iy = (int)fy;
			auto c00 = coeff[ix + iy * coeffPitch];
			auto c01 = coeff[(ix + 1) + iy * coeffPitch];
			auto c10 = coeff[ix + (iy + 1) * coeffPitch];
			auto c11 = coeff[(ix + 1) + (iy + 1) * coeffPitch];
			float fracx = fx - ix;
			float fracy = fy - iy;
			float c = (c00 * (1 - fracx) + c01 * fracx) * (1 - fracy) + (c10 * (1 - fracx) + c11 * fracx) * fracy;
			dst[x + y * pitch] = (int)c;
		}
	}
}

template <typename T> struct TextureFormat { static cudaChannelFormatDesc desc; };

cudaChannelFormatDesc TextureFormat<uint8_t>::desc = { 8, 0, 0, 0, cudaChannelFormatKindUnsigned };
cudaChannelFormatDesc TextureFormat<uint16_t>::desc = { 16, 0, 0, 0, cudaChannelFormatKindUnsigned };

template <typename pixel_t>
cudaResourceDesc makeResourceDesc(
	const pixel_t *ptr, int width, int height, int pitch)
{
	cudaResourceDesc resDesc = cudaResourceDesc();
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = const_cast<pixel_t*>(ptr);
	resDesc.res.pitch2D.desc = TextureFormat<pixel_t>::desc;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = pitch * sizeof(pixel_t);
	return resDesc;
}

cudaTextureDesc makeTextureDesc(
	cudaTextureAddressMode addressMode,
	cudaTextureFilterMode filterMode,
	cudaTextureReadMode readMode)
{
	cudaTextureDesc texDesc = cudaTextureDesc();
	texDesc.addressMode[0] = addressMode;
	texDesc.addressMode[1] = addressMode;
	texDesc.filterMode = filterMode;
	texDesc.readMode = readMode;
	return texDesc;
}

class TextureObject
{
	cudaTextureObject_t obj;
public:
	TextureObject(cudaResourceDesc res, cudaTextureDesc tex, PNeoEnv env) {
		CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &tex, nullptr));
	}
	~TextureObject() {
		cudaDestroyTextureObject(obj);
	}
	operator cudaTextureObject_t() {
		return obj;
	}
};

class SharpenFilter : public KFMFilterBase
{
	PClip qpclip;
	PClip unsharpclip;

	int show;

	VideoInfo coeffvi;

	template <typename pixel_t>
	void ProcPlane(Frame& dst, Frame& src, Frame& unsharp, Frame& qp, Frame& coeff, int plane, PNeoEnv env)
	{
		bool isUV = (plane != PLANAR_Y);
		int shiftx = isUV ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
		int shifty = isUV ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;
		int width = vi.width >> shiftx;
		int height = vi.height >> shifty;
		int qpw = coeffvi.width >> shiftx;
		int qph = coeffvi.height >> shifty;
		
		if (IS_CUDA) {
			{
				// qp -> coeff 変換
				dim3 threads(32, 8);
				dim3 blocks(nblocks(qpw, threads.x), nblocks(qph, threads.y));
				kl_sharpen_coeff << <blocks, threads >> > (
					coeff.GetWritePtr<uint8_t>(plane), qpw, qph, coeff.GetPitch<uint8_t>(plane),
					qp.GetReadPtr<uint16_t>(plane), qp.GetPitch<uint16_t>(plane));
				DEBUG_SYNC;
			}

			{
				TextureObject texSrc(
					makeResourceDesc(src.GetReadPtr<pixel_t>(plane), width, height, src.GetPitch<pixel_t>(plane)),
					makeTextureDesc(cudaAddressModeClamp, cudaFilterModePoint, cudaReadModeElementType), env);
				TextureObject texCoeff(
					makeResourceDesc(coeff.GetReadPtr<uint8_t>(plane), qpw, qph, coeff.GetPitch<uint8_t>(plane)),
					makeTextureDesc(cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeNormalizedFloat), env);

				dim3 threads(32, 8);
				dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));

				if (show) {
					kl_show_sharpen_coeff << <blocks, threads >> > (
						dst.GetWritePtr<pixel_t>(plane), width, height, dst.GetPitch<pixel_t>(plane), texCoeff);
					DEBUG_SYNC;
				}
				else {
					kl_sharpen << <blocks, threads >> > (
						dst.GetWritePtr<pixel_t>(plane), width, height, dst.GetPitch<pixel_t>(plane),
						texSrc, texCoeff, unsharp.GetReadPtr<pixel_t>(plane));
					DEBUG_SYNC;
				}
			}
		}
		else {
			// qp -> coeff 変換
			cpu_sharpen_coeff(
				coeff.GetWritePtr<uint8_t>(plane), qpw, qph, coeff.GetPitch<uint8_t>(plane),
				qp.GetReadPtr<uint16_t>(plane), qp.GetPitch<uint16_t>(plane));

			if (show) {
				cpu_show_sharpen_coeff(
					dst.GetWritePtr<pixel_t>(plane), width, height, dst.GetPitch<pixel_t>(plane),
					coeff.GetReadPtr<uint8_t>(plane), coeff.GetPitch<uint8_t>(plane));
			}
			else {
				cpu_sharpen(
					dst.GetWritePtr<pixel_t>(plane), width, height, dst.GetPitch<pixel_t>(plane),
					src.GetReadPtr<pixel_t>(plane), coeff.GetReadPtr<uint8_t>(plane),
					coeff.GetPitch<uint8_t>(plane), unsharp.GetReadPtr<pixel_t>(plane));
			}
		}
	}

	template <typename pixel_t>
	PVideoFrame GetFrameT(int n, PNeoEnv env)
	{
		typedef typename VectorType<uint8_t>::type vpixel_t;

		Frame src = child->GetFrame(n, env);
		Frame unsharp = unsharpclip->GetFrame(n, env);
		Frame qp = qpclip->GetFrame(n, env);
		Frame coeff = env->NewVideoFrame(coeffvi);
		Frame dst = env->NewVideoFrame(vi);

		if (IS_CUDA) {
			if (!IsAligned(src, vi, env)) {
				env->ThrowError("[SharpenFilter]: source filter returns unaligned frame");
			}
			if (!IsAligned(unsharp, vi, env)) {
				env->ThrowError("[SharpenFilter]: source filter returns unaligned frame");
			}
			if (!IsAligned(qp, vi, env)) {
				env->ThrowError("[SharpenFilter]: source filter returns unaligned frame");
			}
		}

		ProcPlane<pixel_t>(dst, src, unsharp, qp, coeff, PLANAR_Y, env);
		ProcPlane<pixel_t>(dst, src, unsharp, qp, coeff, PLANAR_U, env);
		ProcPlane<pixel_t>(dst, src, unsharp, qp, coeff, PLANAR_V, env);

		return dst.frame;
	}

public:
	SharpenFilter(PClip source, PClip qpclip, PClip unsharp, bool show, IScriptEnvironment* env)
		: KFMFilterBase(source)
		, qpclip(qpclip)
		, unsharpclip(unsharp)
		, show(show)
	{
		if (vi.width & 7) env->ThrowError("[SharpenFilter]: width must be multiple of 8");

		coeffvi = qpclip->GetVideoInfo();
		coeffvi.pixel_type = Get8BitType(coeffvi);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		PNeoEnv env = env_;

		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			return GetFrameT<uint8_t>(n, env);
		case 2:
			return GetFrameT<uint16_t>(n, env);
		default:
			env->ThrowError("[SharpenFilter] Unsupported pixel format");
		}

		return PVideoFrame();
	}

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_MTMODE) {
			return MT_NICE_FILTER;
		}
		return KFMFilterBase::SetCacheHints(cachehints, frame_range);
	}
};

class KDeblock : public KFMFilterBase
{
  PClip qpclip;

  int quality;
  float strength;
  float qp_thresh;
  int show;

  float thresh_a;
  float thresh_b;

  template <typename pixel_t>
  void DeblockPlane(
    int width, int height,
    pixel_t* dst, int dstPitch, const pixel_t* src, int srcPitch,
    const uint16_t* qpTmp, int qpTmpPitch, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    VideoInfo padvi = vi;
		// 有効なデータは外周8ピクセルまでだが、
		// 8の倍数でないときにさらにその外までアクセスが有り得るので
		// アクセス可能な状態にしておく。8ピクセルより外側はアクセスがあったとしても
		// 最終フレームには算入されないので、padしなくて良い
    padvi.width = (width + 7 + 8 * 2) & ~7;
    padvi.height = (height + 7 + 8 * 2) & ~7;
		padvi.pixel_type = GetYType(vi);
    Frame pad = env->NewVideoFrame(padvi);

    Copy(pad.GetWritePtr<pixel_t>() + 8 + 8 * pad.GetPitch<pixel_t>(),
      pad.GetPitch<pixel_t>(), src, srcPitch, width, height, env);

    if (IS_CUDA) {
      kl_padv<vpixel_t> << <dim3(nblocks(width >> 2, 32)), dim3(32, 8) >> > (
        pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
        width >> 2, height, pad.GetPitch<vpixel_t>(), 8);
      DEBUG_SYNC;
      kl_padh<pixel_t> << <dim3(1, nblocks(height + 8 * 2, 32)), dim3(8, 32) >> > (
        pad.GetWritePtr<pixel_t>() + 8, width, height + 8 * 2,
        pad.GetPitch<pixel_t>(), 8);
      DEBUG_SYNC;
    }
    else {
      cpu_padv<vpixel_t>(
        pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
        width >> 2, height, pad.GetPitch<vpixel_t>(), 8);
      cpu_padh<pixel_t>(
        pad.GetWritePtr<pixel_t>() + 8, width, height + 8 * 2,
        pad.GetPitch<pixel_t>(), 8);
    }

		// qpclipのviはchromaも考慮して大き目になっているので、そのままでは使えないことに注意
		VideoInfo qpvi = vi;
		qpvi.width = (width + 7 + 8) >> 3;
		qpvi.height = (height + 7 + 8) >> 3;
		qpvi.pixel_type = VideoInfo::CS_Y16;

    int bits = vi.BitsPerComponent();
    int deblockShift = max(0, quality + bits - 10);
    int deblockMaxV = (1 << (bits + 6 - deblockShift)) - 1;
    int mergeShift = quality + 6 - deblockShift;
    int mergeMaxV = (1 << bits) - 1;
    int count = 1 << quality;

    if (show == 2) {
      if (IS_CUDA) {
        dim3 threads(8, 8);
        dim3 blocks(qpvi.width, qpvi.height);
        kl_deblock_show << <blocks, threads >> > (
          dst, dstPitch, width, height, qpvi.width, qpvi.height,
					qpTmp, qpTmpPitch, thresh_a, thresh_b);
        DEBUG_SYNC;
      }
      else {
        cpu_deblock_show(dst, dstPitch, width, height, qpvi.width, qpvi.height,
					qpTmp, qpTmpPitch, thresh_a, thresh_b);
      }
    }
    else if (!IS_CUDA && IsAVX2Available()) {
      //if(false) {
        // CPUでAVX2が使えるならAVX2版
      VideoInfo tmpvi = vi;
      tmpvi.width = (width + 7 + 8 * 2) & ~7;
      tmpvi.height = (height + 7 + 8 * 2) & ~7;
      tmpvi.pixel_type = VideoInfo::CS_Y16;
      Frame tmpOut = env->NewVideoFrame(tmpvi);

      cpu_deblock_avx(pad.GetReadPtr<pixel_t>(), pad.GetPitch<pixel_t>(),
        qpvi.width, qpvi.height,
        tmpOut.GetWritePtr<uint16_t>(), tmpOut.GetPitch<uint16_t>(),
				qpTmp, qpTmpPitch, count - 1,
        deblockShift, deblockMaxV, strength, thresh_a, thresh_b,
        width, height, dst, dstPitch, mergeShift, mergeMaxV);
    }
    else {
      VideoInfo tmpvi = vi;
      tmpvi.width = (width + 7 + 8 * 2) & ~7;
      tmpvi.height = (height + 7 + 8 * 2) & ~7;
      tmpvi.height *= 4;
      tmpvi.pixel_type = VideoInfo::CS_Y16;
      Frame tmpOut = env->NewVideoFrame(tmpvi);

      if (IS_CUDA) {
        dim3 threads(8, count);
        dim3 blocks(qpvi.width, qpvi.height);
        kl_deblock << <blocks, threads, sizeof(float) * 72 * count >> > (
          pad.GetReadPtr<pixel_t>(), pad.GetPitch<pixel_t>(),
          qpvi.width, qpvi.height,
          tmpOut.GetWritePtr<ushort2>(), tmpOut.GetPitch<ushort2>(),
					qpTmp, qpTmpPitch, count - 1,
          deblockShift, deblockMaxV, strength, thresh_a, thresh_b);
        DEBUG_SYNC;
      }
      else {
        cpu_deblock(pad.GetReadPtr<pixel_t>(), pad.GetPitch<pixel_t>(),
          qpvi.width, qpvi.height,
          tmpOut.GetWritePtr<uint16_t>(), tmpOut.GetPitch<uint16_t>(),
					qpTmp, qpTmpPitch, count - 1,
          deblockShift, deblockMaxV, strength, thresh_a, thresh_b);
      }

      if (IS_CUDA) {
        dim3 threads(32, 8);
        dim3 blocks(nblocks(width >> 2, threads.x), nblocks(height, threads.y));
        kl_merge_deblock << <blocks, threads >> > (width >> 2, height,
          tmpOut.GetReadPtr<ushort4>() + 2 + 8 * tmpOut.GetPitch<ushort4>(),
          tmpOut.GetPitch<ushort4>(), qpvi.height * 8,
          (vpixel_t*)dst, dstPitch >> 2, mergeShift, (float)mergeMaxV);
        DEBUG_SYNC;
      }
      else {
        cpu_merge_deblock(width >> 2, height,
          tmpOut.GetReadPtr<ushort4>() + 2 + 8 * tmpOut.GetPitch<ushort4>(),
          tmpOut.GetPitch<ushort4>(), qpvi.height * 8,
          (vpixel_t*)dst, dstPitch >> 2, mergeShift, (float)mergeMaxV);
      }
    }
  }

	template <typename pixel_t>
	void ShowQPFlag(Frame& dst, int flag, PNeoEnv env) {
		auto getMessage = [](int flag) {
			switch (flag) {
			case QP_TABLE_NONE:
				return "Disabled";
			case QP_TABLE_ONLY:
				return "Using QP table";
			case QP_TABLE_CONSTANT:
				return "Constant QP";
			case QP_TABLE_USING_DC:
				return "Using QP table with DC";
			default:
				return "???";
			}
		};
		char buf[100];
		sprintf_s(buf, "KDeblock: %s", getMessage(flag));
		DrawText<pixel_t>(dst.frame, vi.BitsPerComponent(), 0, 0, buf, env);
	}

  template <typename pixel_t>
  PVideoFrame DeblockEntry(Frame& src, Frame& qp, PNeoEnv env)
  {
		int qpflag = (int)qp.GetProperty("DEBLOCK_QP_FLAG")->GetInt();

		if (qpflag == QP_TABLE_NONE) {
			// QPテーブルがない
			if (show) {
				ShowQPFlag<pixel_t>(src, qpflag, env);
			}
			return src.frame;
		}

		Frame dst = env->NewVideoFrame(vi);

    DeblockPlane(vi.width, vi.height,
      dst.GetWritePtr<pixel_t>(PLANAR_Y), dst.GetPitch<pixel_t>(PLANAR_Y),
			src.GetReadPtr<pixel_t>(PLANAR_Y), src.GetPitch<pixel_t>(PLANAR_Y),
			qp.GetReadPtr<uint16_t>(PLANAR_Y), qp.GetPitch<uint16_t>(PLANAR_Y), env);

    DeblockPlane(vi.width >> logUVx, vi.height >> logUVy,
      dst.GetWritePtr<pixel_t>(PLANAR_U), dst.GetPitch<pixel_t>(PLANAR_U),
			src.GetReadPtr<pixel_t>(PLANAR_U), src.GetPitch<pixel_t>(PLANAR_U),
			qp.GetReadPtr<uint16_t>(PLANAR_U), qp.GetPitch<uint16_t>(PLANAR_U), env);

    DeblockPlane(vi.width >> logUVx, vi.height >> logUVy,
      dst.GetWritePtr<pixel_t>(PLANAR_V), dst.GetPitch<pixel_t>(PLANAR_V),
			src.GetReadPtr<pixel_t>(PLANAR_V), src.GetPitch<pixel_t>(PLANAR_V),
			qp.GetReadPtr<uint16_t>(PLANAR_V), qp.GetPitch<uint16_t>(PLANAR_V), env);

    if (show == 1) {
			ShowQPFlag<pixel_t>(dst, qpflag, env);
    }

    return dst.frame;
  }

  PVideoFrame GetQPFrame(int n, PNeoEnv env)
  {
    if (qpclip) {
      return qpclip->GetFrame(n, env);
    }
    return child->GetFrame(n, env);
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame src = child->GetFrame(n, env);
		Frame qp = qpclip->GetFrame(n, env);
		return DeblockEntry<pixel_t>(src, qp, env);
  }

public:
  KDeblock(PClip source, int quality, float strength, float qp_thresh, 
    PClip qpclip, int show, IScriptEnvironment* env)
    : KFMFilterBase(source)
		, qpclip(qpclip)
    , quality(quality)
    , strength(strength)
    , qp_thresh(qp_thresh)
    , show(show)
  {
    if (vi.width & 7) env->ThrowError("[KDeblock]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KDeblock]: height must be multiple of 8");

    const float W = 5;
    thresh_a = (qp_thresh + W) / (2 * W);
    thresh_b = (W * W - qp_thresh * qp_thresh) / (2 * W);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KDeblock] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
		// 1: 左上に情報を表示
		// 2: デブロッキング領域を表示
		// 3: シャープ化領域を表示
		int show = args[9].AsInt(0);

		bool b_adap = args[7].AsBool(true);
		PClip qpclip = new QPForDeblock(
			args[0].AsClip(),
			(float)args[4].AsFloat(b_adap ? 2.0f : 0.5f), // b_ratio
			args[5].AsClip(),                             // qpclip
			args[6].AsInt(-1),                            // force_qp
			b_adap,                                       // b_adap
			env);

		PClip clip = new KDeblock(
			args[0].AsClip(),          // source
			args[1].AsInt(3),          // quality
			(float)args[2].AsFloat(0), // strength
			(float)args[3].AsFloat(0), // thresh
			qpclip,                    // qpclip
			show,                      // show
			env
		);

		if (args[8].AsBool(false)) { // sharpen
			VideoInfo vi = clip->GetVideoInfo();
			AVSValue args[] = { clip, vi.width, vi.height, 0, 0, vi.width + 0.0001, vi.height + 0.0001, 2 };
			PClip unsharp = env->Invoke("GaussResize", AVSValue(args, 8)).AsClip();
			clip = new SharpenFilter(clip, qpclip, unsharp, show == 3, env);
		}

		return clip;
  }
};

__global__ void kl_scale_qp(int width, int height,
  uint8_t* dst, int dst_pitch, const uint8_t* src, int src_pitch, int scale_type)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < width && y < height) {
    dst[x + y * dst_pitch] = norm_qscale(src[x + y * src_pitch], scale_type);
  }
}

void cpu_scale_qp(int width, int height,
  uint8_t* dst, int dst_pitch, const uint8_t* src, int src_pitch, int scale_type)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * dst_pitch] = norm_qscale(src[x + y * src_pitch], scale_type);
    }
  }
}

// QPテーブルをフレームデータに変換
class ShowQP : public GenericVideoFilter
{
	bool nonB;
	bool dc;
public:
  ShowQP(PClip clip, bool nonB, bool dc, IScriptEnvironment* env_)
    : GenericVideoFilter(clip)
		, nonB(nonB)
		, dc(dc)
  {
    PNeoEnv env = env_;

    vi.pixel_type = VideoInfo::CS_Y8;

    if (vi.sample_type == QPClipInfo::MAGIC_KEY) {
      // ソースはQPClipだった
      auto info = QPClipInfo::GetParam(vi, env);
      vi.width = (info->imageWidth + 15) >> 4;
      vi.height = (info->imageHeight + 15) >> 4;
    }
    else {
      vi.width = (vi.width + 15) >> 4;
      vi.height = (vi.height + 15) >> 4;
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    Frame src = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

		if (dc) {
			const AVSMapValue* prop = src.GetProperty("DC_Table");
			if (prop == nullptr) {
				cpu_fill<uint8_t, 0>(dst.GetWritePtr<uint8_t>(), vi.width, vi.height, dst.GetPitch<uint8_t>());
				DrawText<uint8_t>(dst.frame, 8, 0, 0, "No DC table ...", env);
			}
			else {
				Frame dcTable = prop->GetFrame();
				Copy<uint8_t>(dst.GetWritePtr<uint8_t>(), dst.GetPitch<uint8_t>(), 
					dcTable.GetReadPtr<uint8_t>(), dcTable.GetPitch<uint8_t>(), vi.width, vi.height, env);
			}
		}
		else {
			const AVSMapValue* prop = src.GetProperty(nonB ? "QP_Table_Non_B" : "QP_Table");
			if (prop == nullptr) {
				cpu_fill<uint8_t, 0>(dst.GetWritePtr<uint8_t>(), vi.width, vi.height, dst.GetPitch<uint8_t>());
				DrawText<uint8_t>(dst.frame, 8, 0, 0, "No QP table ...", env);
			}
			else {
				Frame qpTable = prop->GetFrame();
				int qpStride = (int)src.GetProperty("QP_Stride")->GetInt();
				int qpScaleType = (int)src.GetProperty("QP_ScaleType")->GetInt();
				if (IS_CUDA) {
					dim3 threads(32, 8);
					dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
					kl_scale_qp << <blocks, threads >> > (vi.width, vi.height,
						dst.GetWritePtr<uint8_t>(), dst.GetPitch<uint8_t>(),
						qpTable.GetReadPtr<uint8_t>(), qpStride, qpScaleType);
					DEBUG_SYNC;
				}
				else {
					cpu_scale_qp(vi.width, vi.height,
						dst.GetWritePtr<uint8_t>(), dst.GetPitch<uint8_t>(),
						qpTable.GetReadPtr<uint8_t>(), qpStride, qpScaleType);
				}
			}
		}


    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new ShowQP(
      args[0].AsClip(),      // clip
			args[1].AsBool(true),  // nonB
			args[2].AsBool(false), // dc
      env
    );
  }
};

static AVSValue __cdecl FrameType(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  AVSValue clip = args[0];
  PClip child = clip.AsClip();
  VideoInfo vi = child->GetVideoInfo();

  AVSValue cn = env->GetVarDef("current_frame");
  if (!cn.IsInt())
    env->ThrowError("FrameType: This filter can only be used within run-time filters");

  int n = cn.AsInt();
  n = min(max(n, 0), vi.num_frames - 1);

  auto entry = child->GetFrame(n, env)->GetProperty("FrameType");
  if (entry) {
    return (int)entry->GetInt();
  }
  return 0;
}

void AddFuncDeblock(IScriptEnvironment* env)
{
  env->AddFunction("KDeblock", "c[quality]i[str]f[thr]f[bratio]f[qpclip]c[qp]i[badap]b[sharp]b[show]i", KDeblock::Create, 0);
  env->AddFunction("QPClip", "c", QPClip::Create, 0);
  env->AddFunction("ShowQP", "c[nonb]b[dc]b", ShowQP::Create, 0);

  env->AddFunction("FrameType", "c", FrameType, 0);
}
