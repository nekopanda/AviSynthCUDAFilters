
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "Copy.h"
#include "VectorFunctions.cuh"
#include "KFMFilterBase.cuh"

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

__device__ void dev_softthresh(int tx, float *data, float threshold)
{
  for (int i = tx; i < 72; i += 8) {
    if (i == 0) continue;
    float level = data[i];
    if (abs(level) <= threshold) data[i] = 0;
    else if (level > 0) data[i] -= threshold;
    else                data[i] += threshold;
  }
}

__host__ void cpu_hardthresh(float *data, float threshold)
{
	for (int i = 1; i < 64; ++i) {
		float level = data[i];
		if (abs(level) <= threshold) {
			data[i] = 0;
		}
	}
}

__host__ void cpu_softthresh(float *data, float threshold)
{
	for (int i = 1; i < 64; ++i) {
		float level = data[i];
		if (abs(level) <= threshold) data[i] = 0;
		else if (level > 0) data[i] -= threshold;
		else                data[i] += threshold;
	}
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
  float strength, float qp_bias, bool is_soft)
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
	float thresh = (qp + qp_bias) * ((1 << 4) + strength) - 1;
  if (is_soft) {
    dev_softthresh(tx, dct_tmp, thresh);
  }
  else {
    dev_hardthresh(tx, dct_tmp, thresh);
  }
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
  float strength, float qp_bias, bool is_soft)
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

        // dct
				cpu_dct8x8(dct_tmp);

        // requantize
				uint16_t qp = qp_table[bx + by * qp_pitch];
				float thresh = (qp + qp_bias) * ((1 << 4) + strength) - 1;
				if (is_soft) {
					cpu_softthresh(dct_tmp, thresh);
				}
				else {
					cpu_hardthresh(dct_tmp, thresh);
				}
        
				// idct
				cpu_idct8x8(dct_tmp);

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

// normalize the qscale factor
__device__ __host__ inline int norm_qscale(int qscale, int type)
{
  switch (type) {
  case 0/*FF_QSCALE_TYPE_MPEG1*/: return qscale;
  case 1/*FF_QSCALE_TYPE_MPEG2*/: return qscale >> 1;
  case 2/*FF_QSCALE_TYPE_H264*/:  return qscale >> 2;
  case 3/*FF_QSCALE_TYPE_VP56*/:  return (63 - qscale + 2) >> 2;
  }
  return qscale;
}


__global__ void kl_make_qp_table(
  int in_width, int in_height,
  const uint8_t* in_table, int in_pitch, int qp_scale,
	int qp_shift_x, int qp_shift_y,
  int out_width, int out_height,
	uint16_t* out_table, int out_pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < out_width && y < out_height) {
    int qp;
    if (in_table) {
			int qp_x = min(x >> qp_shift_x, in_width - 1);
			int qp_y = min(y >> qp_shift_y, in_height - 1);
      qp = in_table[qp_x + qp_y * in_pitch];
      qp = max(1, norm_qscale(qp, qp_scale));
    }
    else {
      qp = qp_scale;
    }
    out_table[x + y * out_pitch] = qp;
  }
}

void cpu_make_qp_table(
  int in_width, int in_height,
  const uint8_t* in_table, int in_pitch, int qp_scale,
	int qp_shift_x, int qp_shift_y,
  int out_width, int out_height,
  uint16_t* out_table, int out_pitch)
{
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      int qp;
      if (in_table) {
				int qp_x = min(x >> qp_shift_x, in_width - 1);
				int qp_y = min(y >> qp_shift_y, in_height - 1);
				qp = in_table[qp_x + qp_y * in_pitch];
        qp = max(1, norm_qscale(qp, qp_scale));
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
	uint16_t* dst, int dst_pitch, float thresh, bool is_soft, float half, int shift, int maxv);

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
	float strength, float qp_bias, bool is_soft,
	
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
				float thresh = (qp + qp_bias) * ((1 << 4) + strength) - 1;
				const int half = (1 << deblockShift) >> 1;
				const pixel_t* src_block = &src[off_x + off_y * src_pitch];
				uint16_t* tmp_block = &tmp[off_x + off_y * tmp_pitch];

				cpu_deblock_kernel_avx(src_block, src_pitch,
					tmp_block, tmp_pitch, thresh, is_soft, (float)half, deblockShift, deblockMaxV);
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

bool IsAVX2Available();

class KDeblock : public KFMFilterBase
{
	int quality;
	float strength;
	float qp_bias;
	int force_qp;
	bool is_soft;

	template <typename pixel_t>
	void DeblockPlane(
		int width, int height,
		pixel_t* dst, int dstPitch, const pixel_t* src, int srcPitch, 
		const uint8_t* qpTable, int qpStride, int qpScaleType,
		int qpShiftX, int qpShiftY, PNeoEnv env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;

		VideoInfo padvi = vi;
		padvi.width = width + 8 * 2;
		padvi.height = height + 8 * 2;
		Frame pad = env->NewVideoFrame(padvi);

		Copy(pad.GetWritePtr<pixel_t>() + 8 + 8 * pad.GetPitch<pixel_t>(), 
			pad.GetPitch<pixel_t>(), src, srcPitch, width, height, env);

		if (IS_CUDA) {
			kl_padv<vpixel_t> << <dim3(nblocks(width >> 2, 32)), dim3(32, 8) >> > (
				pad.GetWritePtr<vpixel_t>() + 2 + 8 * pad.GetPitch<vpixel_t>(),
				width >> 2, height, pad.GetPitch<vpixel_t>(), 8);
			DEBUG_SYNC;
			kl_padh<pixel_t> << <dim3(1, nblocks(height + 8 * 2, 32)), dim3(8, 32) >> >(
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

		VideoInfo qpvi = vi;
		qpvi.width = (width + 7 + 8) >> 3;
		qpvi.height = (height + 7 + 8) >> 3;
		qpvi.pixel_type = VideoInfo::CS_Y16;
		Frame qpTmp = env->NewVideoFrame(qpvi);

		if (IS_CUDA) {
			dim3 threads(32, 8);
			dim3 blocks(nblocks(qpvi.width, threads.x), nblocks(qpvi.height, threads.y));
			kl_make_qp_table<<<blocks, threads >>>((
				vi.width + 15) >> 4, (vi.height + 15) >> 4,
				qpTable, qpStride, qpTable ? qpScaleType : force_qp,
				qpShiftX, qpShiftY, qpvi.width, qpvi.height,
				qpTmp.GetWritePtr<uint16_t>(), qpTmp.GetPitch<uint16_t>());
			DEBUG_SYNC;
		}
		else {
			cpu_make_qp_table((vi.width + 15) >> 4, (vi.height + 15) >> 4,
				qpTable, qpStride, qpTable ? qpScaleType : force_qp,
				qpShiftX, qpShiftY, qpvi.width, qpvi.height,
				qpTmp.GetWritePtr<uint16_t>(), qpTmp.GetPitch<uint16_t>());
		}

		int bits = vi.BitsPerComponent();
		int deblockShift = max(0, quality + bits - 10);
		int deblockMaxV = (1 << (bits + 6 - deblockShift)) - 1;
		int mergeShift = quality + 6 - deblockShift;
		int mergeMaxV = (1 << bits) - 1;
		int count = 1 << quality;

		if (!IS_CUDA && IsAVX2Available()) {
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
				qpTmp.GetReadPtr<uint16_t>(), qpTmp.GetPitch<uint16_t>(), count - 1,
				deblockShift, deblockMaxV, strength, qp_bias, is_soft,
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
				kl_deblock << <blocks, threads, sizeof(float) * 72 * count >> >(
					pad.GetReadPtr<pixel_t>(), pad.GetPitch<pixel_t>(),
					qpvi.width, qpvi.height,
					tmpOut.GetWritePtr<ushort2>(), tmpOut.GetPitch<ushort2>(),
					qpTmp.GetReadPtr<uint16_t>(), qpTmp.GetPitch<uint16_t>(), count - 1,
					deblockShift, deblockMaxV, strength, qp_bias, is_soft);
				DEBUG_SYNC;
			}
			else {
				cpu_deblock(pad.GetReadPtr<pixel_t>(), pad.GetPitch<pixel_t>(),
					qpvi.width, qpvi.height,
					tmpOut.GetWritePtr<uint16_t>(), tmpOut.GetPitch<uint16_t>(),
					qpTmp.GetReadPtr<uint16_t>(), qpTmp.GetPitch<uint16_t>(), count - 1,
					deblockShift, deblockMaxV, strength, qp_bias, is_soft);
			}

			if (IS_CUDA) {
				dim3 threads(32, 8);
				dim3 blocks(nblocks(width >> 2, threads.x), nblocks(height, threads.y));
				kl_merge_deblock << <blocks, threads >> >(width >> 2, height,
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
	PVideoFrame DeblockEntry(Frame& src, const uint8_t* qpTable, int qpStride, int qpScaleType, PNeoEnv env)
	{
		Frame dst = env->NewVideoFrame(vi);

		DeblockPlane(vi.width, vi.height, 
			dst.GetWritePtr<pixel_t>(PLANAR_Y), dst.GetPitch<pixel_t>(PLANAR_Y), 
			src.GetReadPtr<pixel_t>(PLANAR_Y), src.GetPitch<pixel_t>(PLANAR_Y), 
			qpTable, qpStride, qpScaleType, 1, 1, env);

		DeblockPlane(vi.width >> logUVx, vi.height >> logUVy,
			dst.GetWritePtr<pixel_t>(PLANAR_U), dst.GetPitch<pixel_t>(PLANAR_U),
			src.GetReadPtr<pixel_t>(PLANAR_U), src.GetPitch<pixel_t>(PLANAR_U),
			qpTable, qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, env);

		DeblockPlane(vi.width >> logUVx, vi.height >> logUVy,
			dst.GetWritePtr<pixel_t>(PLANAR_V), dst.GetPitch<pixel_t>(PLANAR_V),
			src.GetReadPtr<pixel_t>(PLANAR_V), src.GetPitch<pixel_t>(PLANAR_V),
			qpTable, qpStride, qpScaleType, 1 - logUVx, 1 - logUVy, env);

		return dst.frame;
	}

	template <typename pixel_t>
	PVideoFrame GetFrameT(int n, PNeoEnv env)
	{
		Frame src = child->GetFrame(n, env);
		if (force_qp > 0) {
			// QP指定あり
			return DeblockEntry<pixel_t>(src, nullptr, 0, 0, env);
		}
		if (src.GetProperty("QP_Table_Non_B")) {
			// QPテーブルあり
			Frame qpTable = src.GetProperty("QP_Table_Non_B")->GetFrame();
			int qpStride = (int)src.GetProperty("QP_Stride")->GetInt();
			int qpScaleType = (int)src.GetProperty("QP_ScaleType")->GetInt();
			return DeblockEntry<pixel_t>(src, qpTable.GetReadPtr<uint8_t>(), qpStride, qpScaleType, env);
		}
		return src.frame;
	}

public:
	KDeblock(PClip source, int quality, float strength, float qp_bias, int force_qp, bool is_soft, IScriptEnvironment* env)
		: KFMFilterBase(source)
		, quality(quality)
		, strength(strength)
		, qp_bias(qp_bias)
		, force_qp(force_qp)
		, is_soft(is_soft)
	{
		if (vi.width & 7) env->ThrowError("[KDeblock]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KDeblock]: height must be multiple of 8");
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
		return new KDeblock(
			args[0].AsClip(),          // source
			args[1].AsInt(3),          // quality
			(float)args[2].AsFloat(0), // strength
			(float)args[3].AsFloat(0), // qp_bias
			args[4].AsInt(-1),         // force_qp
			args[5].AsBool(false),     // is_soft
			env
		);
	}
};

void AddFuncDeblock(IScriptEnvironment* env)
{
	env->AddFunction("KDeblock", "c[quality]i[str]f[bias]f[force_qp]i[is_soft]b", KDeblock::Create, 0);
}
