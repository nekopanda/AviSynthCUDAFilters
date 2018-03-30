
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
			CUDA_CHECK(cudaGetLastError()); \
      CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

#define IS_CUDA (env->GetProperty(AEP_DEVICE_TYPE) == DEV_TYPE_CUDA)

template <typename T> struct VectorType {};

template <> struct VectorType<unsigned char> {
  typedef uchar4 type;
};

template <> struct VectorType<unsigned short> {
  typedef ushort4 type;
};

int Get8BitType(VideoInfo& vi) {
  if (vi.Is420()) return VideoInfo::CS_YV12;
  else if (vi.Is422()) return VideoInfo::CS_YV16;
  else if (vi.Is444()) return VideoInfo::CS_YV24;
  // これ以外は知らん
  return VideoInfo::CS_BGR24;
}

template <typename pixel_t, int fill_v>
void cpu_fill(pixel_t* dst, int width, int height, int pitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[x + y * pitch] = VHelper<pixel_t>::make(fill_v);
		}
	}
}

template <typename pixel_t, int fill_v>
__global__ void kl_fill(pixel_t* dst, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		dst[x + y * pitch] = VHelper<pixel_t>::make(fill_v);
	}
}

template <typename pixel_t>
void cpu_copy(pixel_t* dst, const pixel_t* __restrict__ src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_copy(pixel_t* dst, const pixel_t* __restrict__ src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * pitch] = src[x + y * pitch];
  }
}

template <typename pixel_t>
void cpu_average(pixel_t* dst, const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, int width, int height, int pitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			auto tmp = (to_int(src0[x + y * pitch]) + to_int(src1[x + y * pitch])) >> 1;
			dst[x + y * pitch] = VHelper<pixel_t>::cast_to(tmp);
		}
	}
}

template <typename pixel_t>
__global__ void kl_average(pixel_t* dst, const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		auto tmp = (to_int(src0[x + y * pitch]) + to_int(src1[x + y * pitch])) >> 1;
		dst[x + y * pitch] = VHelper<pixel_t>::cast_to(tmp);
	}
}

template <typename pixel_t>
void cpu_padv(pixel_t* dst, int width, int height, int pitch, int vpad)
{
  for (int y = 0; y < vpad; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + (-y - 1) * pitch] = dst[x + (y)* pitch];
      dst[x + (height + y) * pitch] = dst[x + (height - y - 1)* pitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_padv(pixel_t* dst, int width, int height, int pitch, int vpad)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if (x < width) {
    dst[x + (-y - 1) * pitch] = dst[x + (y)* pitch];
    dst[x + (height + y) * pitch] = dst[x + (height - y - 1)* pitch];
  }
}

template <typename pixel_t>
void cpu_padh(pixel_t* dst, int width, int height, int pitch, int hpad)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < hpad; ++x) {
			dst[(-x - 1) + y * pitch] = dst[(x) + y * pitch];
			dst[(width + x) + y * pitch] = dst[(width - x - 1) + y * pitch];
		}
	}
}

template <typename pixel_t>
__global__ void kl_padh(pixel_t* dst, int width, int height, int pitch, int hpad)
{
	int x = threadIdx.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (y < height) {
		dst[(-x - 1) + y * pitch] = dst[(x)+y * pitch];
		dst[(width + x) + y * pitch] = dst[(width - x - 1) + y * pitch];
	}
}

template <typename pixel_t>
void cpu_copy_border(pixel_t* dst,
  const pixel_t* src, int width, int height, int pitch, int vborder)
{
  for (int y = 0; y < vborder; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * pitch] = src[x + y * pitch];
      dst[x + (height - y - 1) * pitch] = src[x + (height - y - 1) * pitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_copy_border(pixel_t* dst,
  const pixel_t* __restrict__ src, int width, int height, int pitch, int vborder)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if (x < width) {
    dst[x + y * pitch] = src[x + y * pitch];
    dst[x + (height - y - 1) * pitch] = src[x + (height - y - 1) * pitch];
  }
}

__device__ __host__ uint8_t MakeDiffFlag(int t, int diff, int threshM, int threshS, int threshLS) {
  uint8_t flag = 0;
  if (t > threshS) flag |= SHIMA;
  if (t > threshLS) flag |= LSHIMA;
  if (diff > threshM) flag |= MOVE;
  return flag;
}

// srefはbase-1ライン
template <typename vpixel_t>
void cpu_analyze_frame(uchar4* dst, int dstPitch,
  const vpixel_t* base, const vpixel_t* sref, const vpixel_t* mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      vpixel_t a = base[x + (y - 1) * pitch];
      vpixel_t b = sref[x + y * pitch];
      vpixel_t c = base[x + y * pitch];
      vpixel_t d = sref[x + (y + 1) * pitch];
      vpixel_t e = base[x + (y + 1) * pitch];
      int4 t = CalcCombe(to_int(a), to_int(b), to_int(c), to_int(d), to_int(e));
      int4 diff = absdiff(mref[x + y * pitch], c);
      uchar4 flags = {
        MakeDiffFlag(t.x, diff.x, threshM, threshS, threshLS),
        MakeDiffFlag(t.y, diff.y, threshM, threshS, threshLS),
        MakeDiffFlag(t.z, diff.z, threshM, threshS, threshLS),
        MakeDiffFlag(t.w, diff.w, threshM, threshS, threshLS),
      };
      // フラグ格納
      dst[x + y * dstPitch] = flags;
    }
  }
}

template <typename vpixel_t>
__global__ void kl_analyze_frame(uchar4* dst, int dstPitch,
  const vpixel_t* __restrict__ base,
  const vpixel_t* __restrict__ sref,
  const vpixel_t* __restrict__ mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    vpixel_t a = base[x + (y - 1) * pitch];
    vpixel_t b = sref[x + y * pitch];
    vpixel_t c = base[x + y * pitch];
    vpixel_t d = sref[x + (y + 1) * pitch];
    vpixel_t e = base[x + (y + 1) * pitch];
    int4 t = CalcCombe(to_int(a), to_int(b), to_int(c), to_int(d), to_int(e));
    int4 diff = absdiff(mref[x + y * pitch], c);
    uchar4 flags = {
      MakeDiffFlag(t.x, diff.x, threshM, threshS, threshLS),
      MakeDiffFlag(t.y, diff.y, threshM, threshS, threshLS),
      MakeDiffFlag(t.z, diff.z, threshM, threshS, threshLS),
      MakeDiffFlag(t.w, diff.w, threshM, threshS, threshLS),
    };
    // フラグ格納
    dst[x + y * dstPitch] = flags;
  }
}

void cpu_merge_uvflags(uint8_t* fY,
  const uint8_t* fU, const uint8_t* fV,
  int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
      int flagUV = fU[offUV] | fV[offUV];
      fY[x + y * pitchY] |= (flagUV << 4);
    }
  }
}

__global__ void kl_merge_uvflags(uint8_t* fY,
  const uint8_t* __restrict__ fU, const uint8_t* __restrict__ fV,
  int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
    int flagUV = fU[offUV] | fV[offUV];
    fY[x + y * pitchY] |= (flagUV << 4);
  }
}

template <typename pixel_t>
void cpu_merge_uvcoefs(pixel_t* fY,
  const pixel_t* fU, const pixel_t* fV,
  int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
      fY[x + y * pitchY] = max(fY[x + y * pitchY], max(fU[offUV], fV[offUV]));
    }
  }
}

template <typename pixel_t>
__global__ void kl_merge_uvcoefs(pixel_t* fY,
  const pixel_t* __restrict__ fU, const pixel_t* __restrict__ fV,
  int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
    fY[x + y * pitchY] = max(fY[x + y * pitchY], max(fU[offUV], fV[offUV]));
  }
}

template <typename vpixel_t>
void cpu_and_coefs(vpixel_t* dstp, const vpixel_t* diffp,
  int width, int height, int pitch, float invcombe, float invdiff)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float4 combe = clamp(to_float(dstp[x + y * pitch]) * invcombe + (-1.0f), -0.5f, 0.5f);
      float4 diff = clamp(to_float(diffp[x + y * pitch]) * (-invdiff) + 1.0f, -0.5f, 0.5f);
      float4 tmp = max(combe + diff, 0.0f) * 128.0f + 0.5f;
      dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename pixel_t>
void cpu_apply_uvcoefs_420(
  const pixel_t* fY, pixel_t* fU, pixel_t* fV,
  int widthUV, int heightUV, int pitchY, int pitchUV)
{
  for (int y = 0; y < heightUV; ++y) {
    for (int x = 0; x < widthUV; ++x) {
      int v =
        fY[(x * 2 + 0) + (y * 2 + 0) * pitchY] + fY[(x * 2 + 1) + (y * 2 + 0) * pitchY] +
        fY[(x * 2 + 0) + (y * 2 + 1) * pitchY] + fY[(x * 2 + 1) + (y * 2 + 1) * pitchY];
      fU[x + y * pitchUV] = fV[x + y * pitchUV] = (v + 2) >> 2;
    }
  }
}

template <typename pixel_t>
__global__ void kl_apply_uvcoefs_420(
  const pixel_t* __restrict__ fY, pixel_t* fU, pixel_t* fV,
  int widthUV, int heightUV, int pitchY, int pitchUV)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < widthUV && y < heightUV) {
    int v =
      fY[(x * 2 + 0) + (y * 2 + 0) * pitchY] + fY[(x * 2 + 1) + (y * 2 + 0) * pitchY] +
      fY[(x * 2 + 0) + (y * 2 + 1) * pitchY] + fY[(x * 2 + 1) + (y * 2 + 1) * pitchY];
    fU[x + y * pitchUV] = fV[x + y * pitchUV] = (v + 2) >> 2;
  }
}

template <typename vpixel_t>
void cpu_extend_coef(vpixel_t* dst, const vpixel_t* src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 tmp = max(to_int(src[x + (y - 1) * pitch]), max(to_int(src[x + y * pitch]), to_int(src[x + (y + 1) * pitch])));
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_extend_coef(vpixel_t* dst, const vpixel_t* __restrict__ src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 tmp = max(to_int(src[x + (y - 1) * pitch]), max(to_int(src[x + y * pitch]), to_int(src[x + (y + 1) * pitch])));
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

__device__ __host__ int4 CalcCombe(int4 a, int4 b, int4 c, int4 d, int4 e) {
  return abs(a + c * 4 + e - (b + d) * 3);
}

template <typename vpixel_t>
void cpu_calc_combe(vpixel_t* dst, const vpixel_t* src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 combe = CalcCombe(
        to_int(src[x + (y - 2) * pitch]),
        to_int(src[x + (y - 1) * pitch]),
        to_int(src[x + (y + 0) * pitch]),
        to_int(src[x + (y + 1) * pitch]),
        to_int(src[x + (y + 2) * pitch]));

      int4 tmp = clamp(combe >> 2, 0, 255);
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_calc_combe(vpixel_t* dst, const vpixel_t* __restrict__ src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 combe = CalcCombe(
      to_int(src[x + (y - 2) * pitch]),
      to_int(src[x + (y - 1) * pitch]),
      to_int(src[x + (y + 0) * pitch]),
      to_int(src[x + (y + 1) * pitch]),
      to_int(src[x + (y + 2) * pitch]));

    int4 tmp = clamp(combe >> 2, 0, 255);
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

struct FrameAnalyzeParam {
  int threshM;
  int threshS;
  int threshLS;

  FrameAnalyzeParam(int M, int S, int LS)
    : threshM(M)
    , threshS(S * 6)
    , threshLS(LS * 6)
  { }
};

class KFMFilterBase : public GenericVideoFilter {
protected:
  VideoInfo srcvi;
  int logUVx;
  int logUVy;

  template <typename pixel_t>
  void CopyFrame(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
    const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = srcvi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = srcvi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_copy << <blocks, threads >> >(dstY, srcY, width4, srcvi.height, pitchY);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstU, srcU, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstV, srcV, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_copy<vpixel_t>(dstY, srcY, width4, srcvi.height, pitchY);
      cpu_copy<vpixel_t>(dstU, srcU, width4UV, heightUV, pitchUV);
      cpu_copy<vpixel_t>(dstV, srcV, width4UV, heightUV, pitchUV);
    }
  }

  template <typename pixel_t>
  void PadFrame(PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = dst->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = dst->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = srcvi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = srcvi.height >> logUVy;
    int vpadUV = VPAD >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, VPAD);
      dim3 blocks(nblocks(width4, threads.x));
      dim3 threadsUV(32, vpadUV);
      dim3 blocksUV(nblocks(width4UV, threads.x));
      kl_padv << <blocks, threads >> >(dstY, width4, srcvi.height, pitchY, VPAD);
      DEBUG_SYNC;
      kl_padv << <blocksUV, threadsUV >> >(dstU, width4UV, heightUV, pitchUV, vpadUV);
      DEBUG_SYNC;
      kl_padv << <blocksUV, threadsUV >> >(dstV, width4UV, heightUV, pitchUV, vpadUV);
      DEBUG_SYNC;
    }
    else {
      cpu_padv<vpixel_t>(dstY, width4, srcvi.height, pitchY, VPAD);
      cpu_padv<vpixel_t>(dstU, width4UV, heightUV, pitchUV, vpadUV);
      cpu_padv<vpixel_t>(dstV, width4UV, heightUV, pitchUV, vpadUV);
    }
  }

  template <typename vpixel_t>
  void LaunchAnalyzeFrame(uchar4* dst, int dstPitch,
    const vpixel_t* base, const vpixel_t* sref, const vpixel_t* mref,
    int width, int height, int pitch, int threshM, int threshS, int threshLS,
    IScriptEnvironment2* env)
  {
    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      kl_analyze_frame << <blocks, threads >> >(
        dst, dstPitch, base, sref, mref, width, height, pitch, threshM, threshS, threshLS);
    }
    else {
      cpu_analyze_frame(
        dst, dstPitch, base, sref, mref, width, height, pitch, threshM, threshS, threshLS);
    }
  }

  template <typename pixel_t>
  void AnalyzeFrame(PVideoFrame& f0, PVideoFrame& f1, PVideoFrame& flag, 
    const FrameAnalyzeParam* prmY, const FrameAnalyzeParam* prmC, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    // 各プレーンを判定
    for (int pi = 0; pi < 3; ++pi) {
      int p = planes[pi];

      const vpixel_t* f0p = reinterpret_cast<const vpixel_t*>(f0->GetReadPtr(p));
      const vpixel_t* f1p = reinterpret_cast<const vpixel_t*>(f1->GetReadPtr(p));
      uchar4* flagp = reinterpret_cast<uchar4*>(flag->GetWritePtr(p));
      int pitch = f0->GetPitch(p) / sizeof(vpixel_t);
      int dstPitch = flag->GetPitch(p) / sizeof(uchar4);

      int width4 = srcvi.width >> 2;
      int height = srcvi.height;
      if (pi > 0) {
        width4 >>= logUVx;
        height >>= logUVy;
      }

      auto prm = (pi == 0) ? prmY : prmC;

      // top
      LaunchAnalyzeFrame(
        flagp, dstPitch * 2,
        f0p,
        f0p - pitch,
        f1p,
        width4, height / 2, pitch * 2,
        prm->threshM, prm->threshS, prm->threshLS, env);

      // bottom
      LaunchAnalyzeFrame(
        flagp + dstPitch, dstPitch * 2,
        f0p + pitch,
        f1p,
        f1p + pitch,
        width4, height / 2, pitch * 2,
        prm->threshM, prm->threshS, prm->threshLS, env);
    }
  }

  void MergeUVFlags(PVideoFrame& flag, IScriptEnvironment2* env)
  {
    uint8_t* fY = reinterpret_cast<uint8_t*>(flag->GetWritePtr(PLANAR_Y));
    uint8_t* fU = reinterpret_cast<uint8_t*>(flag->GetWritePtr(PLANAR_U));
    uint8_t* fV = reinterpret_cast<uint8_t*>(flag->GetWritePtr(PLANAR_V));
    int pitchY = flag->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int pitchUV = flag->GetPitch(PLANAR_U) / sizeof(uint8_t);

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(srcvi.width, threads.x), nblocks(srcvi.height, threads.y));
      kl_merge_uvflags << <blocks, threads >> >(fY,
        fU, fV, srcvi.width, srcvi.height, pitchY, pitchUV, logUVx, logUVy);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_uvflags(fY,
        fU, fV, srcvi.width, srcvi.height, pitchY, pitchUV, logUVx, logUVy);
    }
  }

  template <typename pixel_t>
  void MergeUVCoefs(PVideoFrame& flag, IScriptEnvironment2* env)
  {
    pixel_t* fY = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_Y));
    pixel_t* fU = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_U));
    pixel_t* fV = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_V));
    int pitchY = flag->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = flag->GetPitch(PLANAR_U) / sizeof(pixel_t);

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
      kl_merge_uvcoefs << <blocks, threads >> >(fY,
        fU, fV, vi.width, vi.height, pitchY, pitchUV, logUVx, logUVy);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_uvcoefs(fY,
        fU, fV, vi.width, vi.height, pitchY, pitchUV, logUVx, logUVy);
    }
  }

  template <typename pixel_t>
  void ApplyUVCoefs(PVideoFrame& flag, IScriptEnvironment2* env)
  {
    pixel_t* fY = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_Y));
    pixel_t* fU = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_U));
    pixel_t* fV = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_V));
    int pitchY = flag->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = flag->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_apply_uvcoefs_420 << <blocks, threads >> >(fY,
        fU, fV, widthUV, heightUV, pitchY, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_apply_uvcoefs_420(fY, fU, fV, widthUV, heightUV, pitchY, pitchUV);
    }
  }

  template <typename pixel_t>
  void ExtendCoefs(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      kl_extend_coef<< <blocks, threads >> >(
        dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
      DEBUG_SYNC;
      dim3 threadsB(32, 1);
      dim3 blocksB(nblocks(width4, threads.x));
      kl_copy_border << <blocksB, threadsB >> > (
        dstY, srcY, width4, vi.height, pitchY, 1);
      DEBUG_SYNC;
    }
    else {
      cpu_extend_coef(dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
      cpu_copy_border(dstY, srcY, width4, vi.height, pitchY, 1);
    }
  }

  template <typename pixel_t>
  void CompareFields(PVideoFrame& src, PVideoFrame& flag, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
    const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_calc_combe << <blocks, threads >> >(dstY, srcY, width4, vi.height, pitchY);
      DEBUG_SYNC;
      kl_calc_combe << <blocksUV, threads >> >(dstU, srcU, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_calc_combe << <blocksUV, threads >> >(dstV, srcV, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_calc_combe(dstY, srcY, width4, vi.height, pitchY);
      cpu_calc_combe(dstU, srcU, width4UV, heightUV, pitchUV);
      cpu_calc_combe(dstV, srcV, width4UV, heightUV, pitchUV);
    }
  }

  PVideoFrame OffsetPadFrame(const PVideoFrame& frame, IScriptEnvironment2* env)
  {
    int vpad = VPAD;
    int vpadUV = VPAD >> logUVy;

    return env->SubframePlanar(frame,
      frame->GetPitch(PLANAR_Y) * vpad, frame->GetPitch(PLANAR_Y), frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y) - vpad * 2,
      frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U));
  }

public:
  KFMFilterBase(PClip _child)
    : GenericVideoFilter(_child)
    , srcvi(vi)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  { }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceType(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }
};

template <typename vpixel_t>
void cpu_min_frames(vpixel_t* dst,
  const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2,
  int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 a = to_int(src0[x + y * pitch]);
      int4 b = to_int(src1[x + y * pitch]);
      int4 c = to_int(src2[x + y * pitch]);

      int4 minv = min(a, min(b, c));

      // フラグ格納
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(minv);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_min_frames(vpixel_t* dst,
  const vpixel_t* __restrict__ src0, 
  const vpixel_t* __restrict__ src1, 
  const vpixel_t* __restrict__ src2,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 a = to_int(src0[x + y * pitch]);
    int4 b = to_int(src1[x + y * pitch]);
    int4 c = to_int(src2[x + y * pitch]);

    int4 minv = min(a, min(b, c));

    // フラグ格納
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(minv);
  }
}

template <typename vpixel_t>
void cpu_compare_frames(vpixel_t* dst,
	const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2, const vpixel_t* src3, const vpixel_t* src4,
	int width, int height, int pitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int4 a = to_int(src0[x + y * pitch]);
			int4 b = to_int(src1[x + y * pitch]);
			int4 c = to_int(src2[x + y * pitch]);
			int4 d = to_int(src3[x + y * pitch]);
			int4 e = to_int(src4[x + y * pitch]);

			int4 minv = min(min(a, b), min(c, min(d, e)));
			int4 maxv = max(max(a, b), max(c, max(d, e)));

			// フラグ格納
			dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
		}
	}
}

template <typename vpixel_t>
__global__ void kl_compare_frames(vpixel_t* dst,
	const vpixel_t* __restrict__ src0,
	const vpixel_t* __restrict__ src1,
	const vpixel_t* __restrict__ src2,
	const vpixel_t* __restrict__ src3,
	const vpixel_t* __restrict__ src4,
	int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int4 a = to_int(src0[x + y * pitch]);
		int4 b = to_int(src1[x + y * pitch]);
		int4 c = to_int(src2[x + y * pitch]);
		int4 d = to_int(src3[x + y * pitch]);
		int4 e = to_int(src4[x + y * pitch]);

		int4 minv = min(min(a, b), min(c, min(d, e)));
		int4 maxv = max(max(a, b), max(c, max(d, e)));

		// フラグ格納
		dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
	}
}

template <typename vpixel_t>
__global__ void kl_and_coefs(vpixel_t* dstp, const vpixel_t* __restrict__ diffp,
  int width, int height, int pitch, float invcombe, float invdiff)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    float4 combe = clamp(to_float(dstp[x + y * pitch]) * invcombe + (-1.0f), -0.5f, 0.5f);
    float4 diff = clamp(to_float(diffp[x + y * pitch]) * (-invdiff) + 1.0f, -0.5f, 0.5f);
    float4 tmp = max(combe + diff, 0.0f) * 128.0f + 0.5f;
    dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KTemporalDiff : public KFMFilterBase
{
	enum {
		DIST = 2,
		N_REFS = DIST * 2 + 1,
	};

	PVideoFrame GetRefFrame(int ref, IScriptEnvironment2* env)
	{
		ref = clamp(ref, 0, vi.num_frames);
		return child->GetFrame(ref, env);
	}

	template <typename pixel_t>
	void CompareFrames(PVideoFrame* frames, PVideoFrame& flag, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;

		const vpixel_t* srcY[N_REFS];
		const vpixel_t* srcU[N_REFS];
		const vpixel_t* srcV[N_REFS];

		for (int i = 0; i < N_REFS; ++i) {
			srcY[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_Y));
			srcU[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_U));
			srcV[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_V));
		}

		vpixel_t* dstY = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_Y));
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_V));

		int pitchY = frames[0]->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = frames[0]->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_compare_frames << <blocks, threads >> >(dstY,
				srcY[0], srcY[1], srcY[2], srcY[3], srcY[4], width4, vi.height, pitchY);
			DEBUG_SYNC;
			kl_compare_frames << <blocksUV, threads >> >(dstU,
				srcU[0], srcU[1], srcU[2], srcU[3], srcU[4], width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
			kl_compare_frames << <blocksUV, threads >> >(dstV,
				srcV[0], srcV[1], srcV[2], srcV[3], srcV[4], width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
		}
		else {
			cpu_compare_frames(dstY, srcY[0], srcY[1], srcY[2], srcY[3], srcY[4], width4, vi.height, pitchY);
			cpu_compare_frames(dstU, srcU[0], srcU[1], srcU[2], srcU[3], srcU[4], width4UV, heightUV, pitchUV);
			cpu_compare_frames(dstV, srcV[0], srcV[1], srcV[2], srcV[3], srcV[4], width4UV, heightUV, pitchUV);
		}
	}

	template <typename pixel_t>
	PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
	{
		PVideoFrame frames[N_REFS];
		for (int i = 0; i < N_REFS; ++i) {
			frames[i] = GetRefFrame(i + n - DIST, env);
		}
		PVideoFrame diff = env->NewVideoFrame(vi);

		CompareFrames<pixel_t>(frames, diff, env);

		return diff;
	}

public:
	KTemporalDiff(PClip clip30, IScriptEnvironment2* env)
		: KFMFilterBase(clip30)
	{ }

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			return GetFrameT<uint8_t>(n, env);
		case 2:
			return GetFrameT<uint16_t>(n, env);
		default:
			env->ThrowError("[KTemporalDiff] Unsupported pixel format");
		}

		return PVideoFrame();
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
		return new KTemporalDiff(
			args[0].AsClip(),       // clip30
			env);
	}
};

class KAnalyzeStatic : public KFMFilterBase
{
  enum {
		DIST = 1,
    N_DIFFS = DIST * 2 + 1,
  };

	PClip diffclip;

  VideoInfo padvi;

  float thcombe;
  float thdiff;

  PVideoFrame GetDiffFrame(int ref, IScriptEnvironment2* env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return diffclip->GetFrame(ref, env);
  }

  template <typename pixel_t>
  void GetTemporalDiff(PVideoFrame* frames, PVideoFrame& flag, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    const vpixel_t* srcY[N_DIFFS];
    const vpixel_t* srcU[N_DIFFS];
    const vpixel_t* srcV[N_DIFFS];

    for (int i = 0; i < N_DIFFS; ++i) {
      srcY[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_Y));
      srcU[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_U));
      srcV[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_V));
    }

    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(flag->GetWritePtr(PLANAR_V));

    int pitchY = frames[0]->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = frames[0]->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_min_frames << <blocks, threads >> >(dstY,
				srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
			DEBUG_SYNC;
			kl_min_frames << <blocksUV, threads >> >(dstU,
				srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
			kl_min_frames << <blocksUV, threads >> >(dstV,
				srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
		}
		else {
			cpu_min_frames(dstY, srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
			cpu_min_frames(dstU, srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
			cpu_min_frames(dstV, srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
		}
  }

  template <typename pixel_t>
  void AndCoefs(PVideoFrame& dst, PVideoFrame& flagd, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* diffp = reinterpret_cast<const vpixel_t*>(flagd->GetReadPtr());
    vpixel_t* dstp = reinterpret_cast<vpixel_t*>(dst->GetWritePtr());
    int pitch = dst->GetPitch() / sizeof(vpixel_t);

    // dst: combeありフラグ
    // flagd: diffありフラグ
    float invcombe = 1.0f / thcombe;
    float invdiff = 1.0f / thdiff;
    int width4 = vi.width >> 2;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
      kl_and_coefs << <blocks, threads >> >(
        dstp, diffp, width4, vi.height, pitch, invcombe, invdiff);
      DEBUG_SYNC;
    }
    else {
      cpu_and_coefs(dstp, diffp, width4, vi.height, pitch, invcombe, invdiff);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    PVideoFrame diffframes[N_DIFFS];
    for (int i = 0; i < N_DIFFS; ++i) {
			diffframes[i] = GetDiffFrame(i + n - DIST, env);
    }
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame flagtmp = env->NewVideoFrame(vi);
    PVideoFrame flagc = env->NewVideoFrame(vi);
    PVideoFrame flagd = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(src, padded, env);
    PadFrame<pixel_t>(padded, env);
    CompareFields<pixel_t>(padded, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagc, env);

		GetTemporalDiff<pixel_t>(diffframes, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagd, env);

    AndCoefs<pixel_t>(flagc, flagd, env); // combeありdiffなし -> flagc
    ApplyUVCoefs<pixel_t>(flagc, env);

    return flagc;
  }

public:
  KAnalyzeStatic(PClip clip30, PClip diffclip, float thcombe, float thdiff, IScriptEnvironment2* env)
    : KFMFilterBase(clip30)
		, diffclip(diffclip)
    , thcombe(thcombe)
    , thdiff(thdiff)
    , padvi(vi)
  {
    if (logUVx != 1 || logUVy != 1) env->ThrowError("[KAnalyzeStatic] Unsupported format (only supports YV12)");

    padvi.height += VPAD * 2;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KAnalyzeStatic] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
		PClip clip30 = args[0].AsClip();
		PClip diffclip = env_->Invoke("KTemporalDiff", clip30).AsClip();
    return new KAnalyzeStatic(
			clip30,       // clip30
			diffclip,     // 
      (float)args[1].AsFloat(30),     // thcombe
      (float)args[2].AsFloat(15),     // thdiff
      env);
  }
};

template <typename vpixel_t>
void cpu_merge_static(
  vpixel_t* dstp, const vpixel_t* src60, const vpixel_t* src30, int pitch,
  const vpixel_t* flagp, int width, int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 coef = to_int(flagp[x + y * pitch]);
      int4 v30 = to_int(src30[x + y * pitch]);
      int4 v60 = to_int(src60[x + y * pitch]);
      int4 tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
      dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_merge_static(
  vpixel_t* dstp, const vpixel_t* src60, const vpixel_t* src30, int pitch,
  const vpixel_t* flagp, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 coef = to_int(flagp[x + y * pitch]);
    int4 v30 = to_int(src30[x + y * pitch]);
    int4 v60 = to_int(src60[x + y * pitch]);
    int4 tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
    dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KMergeStatic : public KFMFilterBase
{
  PClip clip30;
  PClip sttclip;

  template <typename pixel_t>
  void MergeStatic(PVideoFrame& src60, PVideoFrame& src30, PVideoFrame& flag, PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* src60Y = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_Y));
    const vpixel_t* src60U = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_U));
    const vpixel_t* src60V = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_V));
    const vpixel_t* src30Y = reinterpret_cast<const vpixel_t*>(src30->GetReadPtr(PLANAR_Y));
    const vpixel_t* src30U = reinterpret_cast<const vpixel_t*>(src30->GetReadPtr(PLANAR_U));
    const vpixel_t* src30V = reinterpret_cast<const vpixel_t*>(src30->GetReadPtr(PLANAR_V));
    const vpixel_t* flagY = reinterpret_cast<const vpixel_t*>(flag->GetReadPtr(PLANAR_Y));
    const vpixel_t* flagU = reinterpret_cast<const vpixel_t*>(flag->GetReadPtr(PLANAR_U));
    const vpixel_t* flagV = reinterpret_cast<const vpixel_t*>(flag->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src60->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src60->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_merge_static << <blocks, threads >> >(
        dstY, src60Y, src30Y, pitchY, flagY, width4, vi.height);
      DEBUG_SYNC;
      kl_merge_static << <blocksUV, threads >> >(
        dstU, src60U, src30U, pitchUV, flagU, width4UV, heightUV);
      DEBUG_SYNC;
      kl_merge_static << <blocksUV, threads >> >(
        dstV, src60V, src30V, pitchUV, flagV, width4UV, heightUV);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_static(dstY, src60Y, src30Y, pitchY, flagY, width4, vi.height);
      cpu_merge_static(dstU, src60U, src30U, pitchUV, flagU, width4UV, heightUV);
      cpu_merge_static(dstV, src60V, src30V, pitchUV, flagV, width4UV, heightUV);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    int n30 = n >> 1;
    PVideoFrame flag = sttclip->GetFrame(n30, env);
    PVideoFrame frame60 = child->GetFrame(n, env);
    PVideoFrame frame30 = clip30->GetFrame(n30, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(frame60, dst, env);
    MergeStatic<pixel_t>(frame60, frame30, flag, dst, env);

    return dst;
  }

public:
  KMergeStatic(PClip clip60, PClip clip30, PClip sttclip, IScriptEnvironment2* env)
    : KFMFilterBase(clip60)
    , clip30(clip30)
    , sttclip(sttclip)
  {
    VideoInfo srcvi = clip30->GetVideoInfo();
    if (vi.num_frames != srcvi.num_frames * 2) {
      env->ThrowError("[KMergeStatic] Num frames don't match");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KMergeStatic] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    return new KMergeStatic(
      args[0].AsClip(),       // clip60
      args[1].AsClip(),       // clip30
      args[2].AsClip(),       // sttclip
      env);
  }
};

__device__ __host__ void CountFlag(FMCount& cnt, int flag)
{
  if (flag & MOVE) cnt.move++;
  if (flag & SHIMA) cnt.shima++;
  if (flag & LSHIMA) cnt.lshima++;
}

void cpu_count_fmflags(FMCount* dst, const uchar4* flagp, int width, int height, int pitch)
{
  FMCount cnt = { 0 };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uchar4 flags = flagp[x + y * pitch];
      CountFlag(cnt, flags.x);
      CountFlag(cnt, flags.y);
      CountFlag(cnt, flags.z);
      CountFlag(cnt, flags.w);
    }
  }
  *dst = cnt;
}

__global__ void kl_init_fmcount(FMCount* dst)
{
  int tx = threadIdx.x;
  dst[tx].move = dst[tx].shima = dst[tx].lshima = 0;
}

enum {
  FM_COUNT_TH_W = 32,
  FM_COUNT_TH_H = 16,
  FM_COUNT_THREADS = FM_COUNT_TH_W * FM_COUNT_TH_H,
};

__global__ void kl_count_fmflags(FMCount* dst, const uchar4* flagp, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * FM_COUNT_TH_W;
  int y = threadIdx.y + blockIdx.y * FM_COUNT_TH_H;
  int tid = threadIdx.x + threadIdx.y * FM_COUNT_TH_W;

  __shared__ int sbuf[FM_COUNT_THREADS];

  FMCount cnt = { 0 };

  if (x < width && y < height) {
    uchar4 flags = flagp[x + y * pitch];
    CountFlag(cnt, flags.x);
    CountFlag(cnt, flags.y);
    CountFlag(cnt, flags.z);
    CountFlag(cnt, flags.w);
  }

  dev_reduce<int, FM_COUNT_THREADS, AddReducer<int>>(tid, cnt.move, sbuf);
  __syncthreads();
  dev_reduce<int, FM_COUNT_THREADS, AddReducer<int>>(tid, cnt.shima, sbuf);
  __syncthreads();
  dev_reduce<int, FM_COUNT_THREADS, AddReducer<int>>(tid, cnt.lshima, sbuf);
  __syncthreads();

  if (tid == 0) {
    atomicAdd(&dst->move, cnt.move);
    atomicAdd(&dst->shima, cnt.shima);
    atomicAdd(&dst->lshima, cnt.lshima);
  }
}

class KFMFrameAnalyze : public KFMFilterBase
{
  VideoInfo padvi;
  VideoInfo flagvi;

  FrameAnalyzeParam prmY;
  FrameAnalyzeParam prmC;

  void CountFlags(PVideoFrame& flag, PVideoFrame& dst, int parity, IScriptEnvironment2* env)
  {
    const uchar4* flagp = reinterpret_cast<const uchar4*>(flag->GetReadPtr(PLANAR_Y));
    FMCount* fmcnt = reinterpret_cast<FMCount*>(dst->GetWritePtr());
    int width4 = srcvi.width >> 2;
    int flagPitch = flag->GetPitch(PLANAR_Y) / sizeof(uchar4);

    FMCount* fmcnt0 = &fmcnt[0];
    FMCount* fmcnt1 = &fmcnt[1];
    if (!parity) {
      std::swap(fmcnt0, fmcnt1);
    }

    if (IS_CUDA) {
      dim3 threads(FM_COUNT_TH_W, FM_COUNT_TH_H);
      dim3 blocks(nblocks(srcvi.width, threads.x), nblocks(srcvi.height / 2, threads.y));
      kl_init_fmcount << <1, 2 >> > (fmcnt);
      DEBUG_SYNC;
      kl_count_fmflags << <blocks, threads >> >(
        fmcnt0, flagp, width4, srcvi.height / 2, flagPitch * 2);
      DEBUG_SYNC;
      kl_count_fmflags << <blocks, threads >> >(
        fmcnt1, flagp + flagPitch, width4, srcvi.height / 2, flagPitch * 2);
      DEBUG_SYNC;
    }
    else {
      cpu_count_fmflags(fmcnt0, flagp, width4, srcvi.height / 2, flagPitch * 2);
      cpu_count_fmflags(fmcnt1, flagp + flagPitch, width4, srcvi.height / 2, flagPitch * 2);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    int parity = child->GetParity(n);
    PVideoFrame f0 = child->GetFrame(n, env);
    PVideoFrame f1 = child->GetFrame(n + 1, env);
    PVideoFrame f0padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame f1padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame fflag = env->NewVideoFrame(flagvi);
    PVideoFrame dst = env->NewVideoFrame(vi);

    // TODO: 切り出し
    CopyFrame<pixel_t>(f0, f0padded, env);
    PadFrame<pixel_t>(f0padded, env);
    CopyFrame<pixel_t>(f1, f1padded, env);
    PadFrame<pixel_t>(f1padded, env);

    AnalyzeFrame<pixel_t>(f0padded, f1padded, fflag, &prmY, &prmC, env);
    MergeUVFlags(fflag, env); // UV判定結果をYにマージ
    CountFlags(fflag, dst, parity, env);

    return dst;
  }

public:
  KFMFrameAnalyze(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
    , padvi(vi)
    , flagvi()
  {
    padvi.height += VPAD * 2;

    int out_bytes = sizeof(FMCount) * 2;
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);

    flagvi.pixel_type = Get8BitType(srcvi);
    flagvi.width = srcvi.width;
    flagvi.height = srcvi.height;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KFMFrameAnalyze] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyze(
      args[0].AsClip(),       // clip
      args[1].AsInt(15),      // threshMY
      args[2].AsInt(7),       // threshSY
      args[3].AsInt(20),      // threshMC
      args[4].AsInt(8),       // threshSC
      env
    );
  }
};

class KFMFrameAnalyzeCheck : public GenericVideoFilter
{
  PClip clipB;
public:
  KFMFrameAnalyzeCheck(PClip clipA, PClip clipB, IScriptEnvironment* env)
    : GenericVideoFilter(clipA)
    , clipB(clipB)
  {}

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    
    PVideoFrame frameA = child->GetFrame(n, env);
    PVideoFrame frameB = clipB->GetFrame(n, env);

    const FMCount* fmcntA = reinterpret_cast<const FMCount*>(frameA->GetReadPtr());
    const FMCount* fmcntB = reinterpret_cast<const FMCount*>(frameB->GetReadPtr());

    if (memcmp(fmcntA, fmcntB, sizeof(FMCount) * 2)) {
      env->ThrowError("[KFMFrameAnalyzeCheck] Unmatch !!!");
    }

    return frameA;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyzeCheck(
      args[0].AsClip(),       // clipA
      args[1].AsClip(),       // clipB
      env
    );
  }
};

class KFMFrameAnalyzeShow : public KFMFilterBase
{
  typedef uint8_t pixel_t;

  VideoInfo padvi;
  VideoInfo flagvi;

  FrameAnalyzeParam prmY;
  FrameAnalyzeParam prmC;

  int threshMY;
  int threshSY;
  int threshLSY;
  int threshMC;
  int threshSC;
  int threshLSC;

  int logUVx;
  int logUVy;

  void VisualizeFlags(PVideoFrame& dst, PVideoFrame& fflag, IScriptEnvironment2* env)
  {
    // 判定結果を表示
    int black[] = { 0, 128, 128 };
    int blue[] = { 73, 230, 111 };
    int gray[] = { 140, 128, 128 };
    int purple[] = { 197, 160, 122 };

    const pixel_t* fflagp = reinterpret_cast<const pixel_t*>(fflag->GetReadPtr(PLANAR_Y));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int flagPitch = fflag->GetPitch(PLANAR_Y);
    int dstPitchY = dst->GetPitch(PLANAR_Y);
    int dstPitchUV = dst->GetPitch(PLANAR_U);

    // 黒で初期化しておく
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int offY = x + y * dstPitchY;
        int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
        dstY[offY] = black[0];
        dstU[offUV] = black[1];
        dstV[offUV] = black[2];
      }
    }

    // 色を付ける
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int flag = fflagp[x + y * flagPitch];
        flag |= (flag >> 4);

        int* color = nullptr;
        if ((flag & MOVE) && (flag & SHIMA)) {
          color = purple;
        }
        else if (flag & MOVE) {
          color = blue;
        }
        else if (flag & SHIMA) {
          color = gray;
        }

        if (color) {
          int offY = x + y * dstPitchY;
          int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
          dstY[offY] = color[0];
          dstU[offUV] = color[1];
          dstV[offUV] = color[2];
        }
      }
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    PVideoFrame f0 = child->GetFrame(n, env);
    PVideoFrame f1 = child->GetFrame(n + 1, env);
    PVideoFrame f0padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame f1padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame fflag = env->NewVideoFrame(flagvi);
    PVideoFrame dst = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(f0, f0padded, env);
    PadFrame<pixel_t>(f0padded, env);
    CopyFrame<pixel_t>(f1, f1padded, env);
    PadFrame<pixel_t>(f1padded, env);

    AnalyzeFrame<pixel_t>(f0padded, f1padded, fflag, &prmY, &prmC, env);
    MergeUVFlags(fflag, env); // UV判定結果をYにマージ
    VisualizeFlags(dst, fflag, env);

    return dst;
  }

public:
  KFMFrameAnalyzeShow(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , padvi(vi)
    , flagvi()
  {
    padvi.height += VPAD * 2;

    flagvi.pixel_type = Get8BitType(srcvi);
    flagvi.width = srcvi.width;
    flagvi.height = srcvi.height;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KFMFrameDev] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyzeShow(
      args[0].AsClip(),       // clip
      args[1].AsInt(10),       // threshMY
      args[2].AsInt(10),       // threshSY
      args[3].AsInt(10),       // threshMC
      args[4].AsInt(10),       // threshSC
      env
    );
  }
};

class KTelecine : public KFMFilterBase
{
  PClip fmclip;
  bool show;

  PulldownPatterns patterns;

	template <typename pixel_t>
	void CopyField(bool top, PVideoFrame* const * frames, const PVideoFrame& dst, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		PVideoFrame& frame0 = *frames[0];
		const vpixel_t* src0Y = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_Y));
		const vpixel_t* src0U = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_U));
		const vpixel_t* src0V = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_V));
		vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = frame0->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = frame0->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;
		
		if (!top) {
			src0Y += pitchY;
			src0U += pitchUV;
			src0V += pitchUV;
			dstY += pitchY;
			dstU += pitchUV;
			dstV += pitchUV;
		}

		if (frames[1] == nullptr) {
			if (IS_CUDA) {
				dim3 threads(32, 16);
				dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
				dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
				kl_copy << <blocks, threads >> >(dstY, src0Y, width4, vi.height / 2, pitchY * 2);
				DEBUG_SYNC;
				kl_copy << <blocksUV, threads >> >(dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
				DEBUG_SYNC;
				kl_copy << <blocksUV, threads >> >(dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
				DEBUG_SYNC;
			}
			else {
				cpu_copy(dstY, src0Y, width4, vi.height / 2, pitchY * 2);
				cpu_copy(dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
				cpu_copy(dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
			}
		}
		else {
			PVideoFrame& frame1 = *frames[1];
			const vpixel_t* src1Y = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_Y));
			const vpixel_t* src1U = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_U));
			const vpixel_t* src1V = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_V));

			if (!top) {
				src1Y += pitchY;
				src1U += pitchUV;
				src1V += pitchUV;
			}

			if (IS_CUDA) {
				dim3 threads(32, 16);
				dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
				dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
				kl_average << <blocks, threads >> >(dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
				DEBUG_SYNC;
				kl_average << <blocksUV, threads >> >(dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
				DEBUG_SYNC;
				kl_average << <blocksUV, threads >> >(dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
				DEBUG_SYNC;
			}
			else {
				cpu_average(dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
				cpu_average(dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
				cpu_average(dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
			}
		}

	}

  template <typename pixel_t>
  PVideoFrame CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, int parity, IScriptEnvironment2* env)
  {
    // fstartは0or1にする
    if (fstart < 0 || fstart >= 2) {
      n += fstart / 2;
      fstart &= 1;
    }

    assert(fstart == 0 || fstart == 1);
    assert(fnum == 2 || fnum == 3 || fnum == 4);

    if (fstart == 0 && fnum == 2) {
      return clip->GetFrame(n, env);
    }
    else {
      PVideoFrame cur = clip->GetFrame(n, env);
      PVideoFrame nxt = clip->GetFrame(n + 1, env);
      PVideoFrame dst = env->NewVideoFrame(vi);

			// 3フィールドのときは重複フィールドを平均化する

			PVideoFrame* srct[2] = { 0 };
			PVideoFrame* srcb[2] = { 0 };
			
			if (parity) {
				srct[0] = &nxt;
				srcb[0] = &cur;
				if (fnum >= 3) {
					if (fstart == 0) {
						srct[1] = &cur;
					}
					else {
						srcb[1] = &nxt;
					}
				}
			}
			else {
				srct[0] = &cur;
				srcb[0] = &nxt;
				if (fnum >= 3) {
					if (fstart == 0) {
						srcb[1] = &cur;
					}
					else {
						srct[1] = &nxt;
					}
				}
			}

			CopyField<pixel_t>(true, srct, dst, env);
			CopyField<pixel_t>(false, srcb, dst, env);

      return dst;
    }
  }

  void DrawInfo(PVideoFrame& dst, int pattern, float cost, int fnum, IScriptEnvironment2* env) {
    env->MakeWritable(&dst);

    char buf[100]; sprintf(buf, "KFM: %d (%.1f) - %d", pattern, cost, fnum);
    DrawText(dst, true, 0, 0, buf);
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    int cycleIndex = n / 4;
    int parity = child->GetParity(cycleIndex * 5);
    PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
    int pattern = (int)fm->GetProps("KFM_Pattern")->GetInt();
    float cost = (float)fm->GetProps("KFM_Cost")->GetFloat();
    Frame24Info frameInfo = patterns.GetFrame24(pattern, n);

    int fstart = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
    PVideoFrame out = CreateWeaveFrame<pixel_t>(child, 0, fstart, frameInfo.numFields, parity, env);

    if (sizeof(pixel_t) == 1 && !IS_CUDA && show) {
      // 8bit CPUにしか対応していない
      DrawInfo(out, pattern, cost, frameInfo.numFields, env);
    }

    return out;
  }

public:
  KTelecine(PClip child, PClip fmclip, bool show, IScriptEnvironment* env)
    : KFMFilterBase(child)
    , fmclip(fmclip)
    , show(show)
  {
    // フレームレート
    vi.MulDivFPS(4, 5);
    vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KTelecine] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceType(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KTelecine(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // fmclip
      args[2].AsBool(false),  // show
      env
    );
  }
};

enum {
	DC_OVERLAP = 4,
	DC_BLOCK_SIZE = 8,
	DC_BLOCK_TH_W = 8,
	DC_BLOCK_TH_H = 8,
};

template <typename pixel_t, typename vpixel_t>
void cpu_detect_combe(pixel_t* flagp, int fpitch,
	const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int shift)
{
	for (int by = 0; by < nBlkY - 1; ++by) {
		for (int bx = 0; bx < nBlkX - 1; ++bx) {
			int sum = 0;
			for (int tx = 0; tx < 2; ++tx) {
				int x = bx * DC_OVERLAP / 4 + tx;
				int y = by * DC_OVERLAP;
				auto L0 = srcp[x + (y + 0) * pitch];
				auto L1 = srcp[x + (y + 1) * pitch];
				auto L2 = srcp[x + (y + 2) * pitch];
				auto L3 = srcp[x + (y + 3) * pitch];
				auto L4 = srcp[x + (y + 4) * pitch];
				auto L5 = srcp[x + (y + 5) * pitch];
				auto L6 = srcp[x + (y + 6) * pitch];
				auto L7 = srcp[x + (y + 7) * pitch];
				int4 diff8 = absdiff(L0, L7);
				int4 diffT = absdiff(L0, L1) + absdiff(L1, L2) + absdiff(L2, L3) + absdiff(L3, L4) + absdiff(L4, L5) + absdiff(L5, L6) + absdiff(L6, L7) - diff8;
				int4 diffE = absdiff(L0, L2) + absdiff(L2, L4) + absdiff(L4, L6) + absdiff(L6, L7) - diff8;
				int4 diffO = absdiff(L0, L1) + absdiff(L1, L3) + absdiff(L3, L5) + absdiff(L5, L7) - diff8;
				int4 score = diffT - diffE - diffO;
				sum += score.x + score.y + score.z + score.w;
			}
			flagp[(bx + 1) + (by + 1) * fpitch] = clamp(sum >> shift, 0, 255);
		}
	}
}

template <typename pixel_t, typename vpixel_t>
__global__ void kl_detect_combe(pixel_t* flagp, int fpitch,
	const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int shift)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x * DC_BLOCK_TH_W + threadIdx.y;
	int by = blockIdx.y * DC_BLOCK_TH_H + threadIdx.z;

	if (bx < nBlkX - 1 && by < nBlkY - 1) {
		int x = bx * DC_OVERLAP / 4 + tx;
		int y = by * DC_OVERLAP;
		auto L0 = srcp[x + (y + 0) * pitch];
		auto L1 = srcp[x + (y + 1) * pitch];
		auto L2 = srcp[x + (y + 2) * pitch];
		auto L3 = srcp[x + (y + 3) * pitch];
		auto L4 = srcp[x + (y + 4) * pitch];
		auto L5 = srcp[x + (y + 5) * pitch];
		auto L6 = srcp[x + (y + 6) * pitch];
		auto L7 = srcp[x + (y + 7) * pitch];
		int4 diff8 = absdiff(L0, L7);
		int4 diffT = absdiff(L0, L1) + absdiff(L1, L2) + absdiff(L2, L3) + absdiff(L3, L4) + absdiff(L4, L5) + absdiff(L5, L6) + absdiff(L6, L7) - diff8;
		int4 diffE = absdiff(L0, L2) + absdiff(L2, L4) + absdiff(L4, L6) + absdiff(L6, L7) - diff8;
		int4 diffO = absdiff(L0, L1) + absdiff(L1, L3) + absdiff(L3, L5) + absdiff(L5, L7) - diff8;
		int4 score = diffT - diffE - diffO;
		int sum = score.x + score.y + score.z + score.w;
#if CUDART_VERSION >= 9000
		sum += __shfl_down_sync(0xffffffff, sum, 1);
#else
		sum += __shfl_down(sum, 1);
#endif
		if (tx == 0) {
			flagp[(bx + 1) + (by + 1) * fpitch] = clamp(sum >> shift, 0, 255);
		}
	}
}

__device__ __host__ int BinomialMerge(int a, int b, int c, int d, int e, int thresh)
{
  int minv = min(a, min(b, min(c, min(d, e))));
  int maxv = max(a, max(b, max(c, max(d, e))));
  if (maxv - minv < thresh) {
    return (b + 2 * c + d + 2) >> 2;
  }
  return c;
}

template <typename pixel_t>
void cpu_remove_combe2(pixel_t* dst, 
	const pixel_t* src, int width, int height, int pitch,
	const pixel_t* combe, int c_pitch, int thcombe, int thdiff)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
			int score = combe[(x >> 2) + (y >> 2) * c_pitch];
      if (score >= thcombe) {
        dst[x + y * pitch] = BinomialMerge(
          src[x + (y - 2) * pitch],
          src[x + (y - 1) * pitch],
          src[x + y * pitch],
          src[x + (y + 1) * pitch],
          src[x + (y + 2) * pitch],
					thdiff);
      }
      else {
        dst[x + y * pitch] = src[x + y * pitch];
      }
    }
  }
}

template <typename pixel_t>
__global__ void kl_remove_combe2(pixel_t* dst,
	const pixel_t* src, int width, int height, int pitch,
	const pixel_t* combe, int c_pitch, int thcombe, int thdiff)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
		int score = combe[(x >> 2) + (y >> 2) * c_pitch];
    if (score >= thcombe) {
      dst[x + y * pitch] = BinomialMerge(
        src[x + (y - 2) * pitch],
        src[x + (y - 1) * pitch],
        src[x + y * pitch],
        src[x + (y + 1) * pitch],
        src[x + (y + 2) * pitch],
				thdiff);
    }
    else {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

void cpu_max_extend_blocks(uint8_t* dstp, int pitch, int nBlkX, int nBlkY)
{
	for (int by = 1; by < nBlkY; ++by) {
		dstp[0 + by * pitch] = dstp[0 + 1 + (by + 0) * nBlkX];
		for (int bx = 1; bx < nBlkX - 1; ++bx) {
			dstp[bx + by * pitch] = max(
				dstp[bx + by * pitch], dstp[bx + 1 + (by + 0) * pitch]);
		}
	}
	for (int bx = 0; bx < nBlkX; ++bx) {
		dstp[bx] = dstp[bx + pitch];
	}
	for (int by = 1; by < nBlkY - 1; ++by) {
		for (int bx = 0; bx < nBlkX; ++bx) {
			dstp[bx + by * pitch] = max(
				dstp[bx + by * pitch], dstp[bx + 0 + (by + 1) * pitch]);
		}
	}
}

__global__ void kl_max_extend_blocks_h(uint8_t* dstp, const uint8_t* srcp, int pitch, int nBlkX, int nBlkY)
{
	int bx = threadIdx.x + blockIdx.x * blockDim.x;
	int by = threadIdx.y + blockIdx.y * blockDim.y;

	if (bx < nBlkX && by < nBlkY) {
		if (bx == nBlkX - 1) {
			// 書き込む予定がないところにソースをコピーする
			dstp[bx + by * pitch] = srcp[bx + by * pitch];
		}
		else if (bx == 0) {
			dstp[bx + by * pitch] = srcp[bx + 1 + (by + 0) * pitch];
		}
		else {
			dstp[bx + by * pitch] = max(
				srcp[bx + 0 + (by + 0) * pitch], srcp[bx + 1 + (by + 0) * pitch]);
		}
	}
}

__global__ void kl_max_extend_blocks_v(uint8_t* dstp, const uint8_t* srcp, int pitch, int nBlkX, int nBlkY)
{
	int bx = threadIdx.x + blockIdx.x * blockDim.x;
	int by = threadIdx.y + blockIdx.y * blockDim.y;

	if (bx < nBlkX && by < nBlkY) {
		if (by == nBlkY - 1) {
			// 書き込む予定がないところにソースをコピーする
			dstp[bx + by * pitch] = srcp[bx + by * pitch];
		}
		else if (by == 0) {
			dstp[bx + by * pitch] = srcp[bx + 0 + (by + 1) * pitch];
		}
		else {
			dstp[bx + by * pitch] = max(
				srcp[bx + 0 + (by + 0) * pitch], srcp[bx + 0 + (by + 1) * pitch]);
		}
	}
}

template <typename pixel_t>
void cpu_combe_to_flag(pixel_t* flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* combe, int cpitch)
{
	for (int y = 0; y < nBlkY; ++y) {
		for (int x = 0; x < nBlkX; ++x) {
			flag[x + y * fpitch] =
				(combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
				 combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
				 combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
				 combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
		}
	}
}

template <typename pixel_t>
__global__ void kl_combe_to_flag(pixel_t* flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* combe, int cpitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		flag[x + y * fpitch] =
			(combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
				combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
				combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
				combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
	}
}

template <typename pixel_t>
void cpu_sum_box3x3(pixel_t* dst, pixel_t* src, int width, int height, int pitch, int maxv)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			 auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
										src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
										src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
			 dst[x + y * pitch] = min(sumv >> 2, maxv); // 適当に1/4する
		}
	}
}

template <typename pixel_t>
__global__ void kl_sum_box3x3(pixel_t* dst, pixel_t* src, int width, int height, int pitch, int maxv)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
			src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
			src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
		dst[x + y * pitch] = min(sumv >> 2, maxv); // 適当に1/4する
	}
}

PVideoFrame NewSwitchFlagFrame(VideoInfo vi, int hpad, int vpad, IScriptEnvironment2* env)
{
	typedef typename VectorType<uint8_t>::type vpixel_t;

	VideoInfo blockpadvi = vi;
	blockpadvi.width = nblocks(vi.width, OVERLAP) + hpad * 2;
	blockpadvi.height = nblocks(vi.height, OVERLAP) + vpad * 2;
	blockpadvi.pixel_type = VideoInfo::CS_Y8;
	PVideoFrame frame = env->NewVideoFrame(blockpadvi);

	// ゼロ初期化
	vpixel_t* flagp = reinterpret_cast<vpixel_t*>(frame->GetWritePtr());
	int pitch = frame->GetPitch() / sizeof(vpixel_t);
	if (IS_CUDA) {
		dim3 threads(32, 8);
		dim3 blocks(nblocks(blockpadvi.width, threads.x), nblocks(blockpadvi.height, threads.y));
		kl_fill<vpixel_t, 0><<<blocks, threads>>>(flagp, blockpadvi.width, blockpadvi.height, pitch);
	}
	else {
		cpu_fill<vpixel_t, 0>(flagp, blockpadvi.width, blockpadvi.height, pitch);
	}

	return env->SubframePlanar(frame,
		hpad * sizeof(uint8_t) + frame->GetPitch(PLANAR_Y) * vpad,
		frame->GetPitch(PLANAR_Y),
		frame->GetRowSize(PLANAR_Y) - hpad * 2 * sizeof(uint8_t),
		frame->GetHeight(PLANAR_Y) - vpad * 2,
		0, 0, 0);
}

class KRemoveCombeCheck : public GenericVideoFilter
{
  PClip clipB;
  int nBlkX, nBlkY;
public:
  KRemoveCombeCheck(PClip clipA, PClip clipB, IScriptEnvironment* env)
    : GenericVideoFilter(clipA)
    , clipB(clipB)
  {
    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    PVideoFrame frameA = child->GetFrame(n, env)->GetProps(COMBE_FLAG_STR)->GetFrame();
    PVideoFrame frameB = clipB->GetFrame(n, env)->GetProps(COMBE_FLAG_STR)->GetFrame();

    const uint8_t* fmcntA = reinterpret_cast<const uint8_t*>(frameA->GetReadPtr());
    const uint8_t* fmcntB = reinterpret_cast<const uint8_t*>(frameB->GetReadPtr());

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        if (fmcntA[bx + by * nBlkX] != fmcntB[bx + by * nBlkX]) {
          env->ThrowError("[KRemoveCombeCheck] Unmatch !!!");
        }
      }
    }

    return frameA;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KRemoveCombeCheck(
      args[0].AsClip(),       // clipA
      args[1].AsClip(),       // clipB
      env
    );
  }
};

class KRemoveCombe : public KFMFilterBase
{
	VideoInfo padvi;
	VideoInfo combvi;
	VideoInfo blockvi;

	float thsmooth;
	float smooth;
	bool detect_uv;
	bool show;
	float thcombe;

	template <typename pixel_t>
	void DetectCombe(PVideoFrame& src, PVideoFrame& combe, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
		const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
		const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
		uint8_t* combeY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
		uint8_t* combeU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
		uint8_t* combeV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));

		int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int fpitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int fpitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
		int widthUV = combvi.width >> logUVx;
		int heightUV = combvi.height >> logUVy;

		int shift = vi.BitsPerComponent() - 8 + 4;

		if (IS_CUDA) {
			dim3 threads(2, DC_BLOCK_TH_W, DC_BLOCK_TH_H);
			dim3 blocks(nblocks(combvi.width, DC_BLOCK_TH_W), nblocks(combvi.height, DC_BLOCK_TH_H));
			dim3 blocksUV(nblocks(widthUV, DC_BLOCK_TH_W), nblocks(heightUV, DC_BLOCK_TH_H));
			kl_detect_combe << <blocks, threads >> >(combeY, fpitchY, srcY, pitchY, combvi.width, combvi.height, shift);
			DEBUG_SYNC;
			if (detect_uv) {
				kl_detect_combe << <blocksUV, threads >> >(combeU, fpitchUV, srcU, pitchUV, widthUV, heightUV, shift);
				DEBUG_SYNC;
				kl_detect_combe << <blocksUV, threads >> >(combeV, fpitchUV, srcV, pitchUV, widthUV, heightUV, shift);
				DEBUG_SYNC;
			}
		}
		else {
			cpu_detect_combe(combeY, fpitchY, srcY, pitchY, combvi.width, combvi.height, shift);
			if (detect_uv) {
				cpu_detect_combe(combeU, fpitchUV, srcU, pitchUV, widthUV, heightUV, shift);
				cpu_detect_combe(combeV, fpitchUV, srcV, pitchUV, widthUV, heightUV, shift);
			}
		}
	}

	void ExtendBlocks(PVideoFrame& dst, PVideoFrame& tmp, IScriptEnvironment2* env)
	{
		uint8_t* tmpY = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_Y));
		uint8_t* tmpU = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_U));
		uint8_t* tmpV = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_V));
		uint8_t* dstY = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_Y));
		uint8_t* dstU = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_U));
		uint8_t* dstV = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = tmp->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int pitchUV = tmp->GetPitch(PLANAR_U) / sizeof(uint8_t);
		int widthUV = combvi.width >> logUVx;
		int heightUV = combvi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(combvi.width, threads.x), nblocks(combvi.height, threads.y));
			dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
			kl_max_extend_blocks_h << <blocks, threads >> >(tmpY, dstY, pitchY, combvi.width, combvi.height);
			kl_max_extend_blocks_v << <blocks, threads >> >(dstY, tmpY, pitchY, combvi.width, combvi.height);
			DEBUG_SYNC;
			if (detect_uv) {
				kl_max_extend_blocks_h << <blocksUV, threads >> > (tmpU, dstU, pitchUV, widthUV, heightUV);
				kl_max_extend_blocks_v << <blocksUV, threads >> > (dstU, tmpU, pitchUV, widthUV, heightUV);
				DEBUG_SYNC;
				kl_max_extend_blocks_h << <blocksUV, threads >> > (tmpV, dstV, pitchUV, widthUV, heightUV);
				kl_max_extend_blocks_v << <blocksUV, threads >> > (dstV, tmpV, pitchUV, widthUV, heightUV);
				DEBUG_SYNC;
			}
		}
		else {
			cpu_max_extend_blocks(dstY, pitchY, combvi.width, combvi.height);
			if (detect_uv) {
				cpu_max_extend_blocks(dstU, pitchUV, widthUV, heightUV);
				cpu_max_extend_blocks(dstV, pitchUV, widthUV, heightUV);
			}
		}
	}

	void MergeUVCoefs(PVideoFrame& combe, IScriptEnvironment2* env)
	{
		uint8_t* fY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
		uint8_t* fU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
		uint8_t* fV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));
		int pitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int pitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(combvi.width, threads.x), nblocks(combvi.height, threads.y));
			kl_merge_uvcoefs << <blocks, threads >> >(fY,
				fU, fV, combvi.width, combvi.height, pitchY, pitchUV, logUVx, logUVy);
			DEBUG_SYNC;
		}
		else {
			cpu_merge_uvcoefs(fY,
				fU, fV, combvi.width, combvi.height, pitchY, pitchUV, logUVx, logUVy);
		}
	}

	void ApplyUVCoefs(PVideoFrame& combe, IScriptEnvironment2* env)
	{
		uint8_t* fY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
		uint8_t* fU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
		uint8_t* fV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));
		int pitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int pitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
		int widthUV = combvi.width >> logUVx;
		int heightUV = combvi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
			kl_apply_uvcoefs_420 << <blocks, threads >> >(fY,
				fU, fV, widthUV, heightUV, pitchY, pitchUV);
			DEBUG_SYNC;
		}
		else {
			cpu_apply_uvcoefs_420(fY, fU, fV, widthUV, heightUV, pitchY, pitchUV);
		}
	}

	template <typename pixel_t>
	void RemoveCombe(PVideoFrame& dst, PVideoFrame& src, PVideoFrame& combe, int thcombe, int thdiff, IScriptEnvironment2* env)
	{
		const uint8_t* combeY = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
		const uint8_t* combeU = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_U));
		const uint8_t* combeV = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_V));
		const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
		const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
		const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
		int fpitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int fpitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
		int widthUV = vi.width >> logUVx;
		int heightUV = vi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
			kl_remove_combe2 << <blocks, threads >> >(dstY, srcY, vi.width, vi.height, pitchY, combeY, fpitchY, thcombe, thdiff);
			DEBUG_SYNC;
			kl_remove_combe2 << <blocksUV, threads >> >(dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thcombe, thdiff);
			DEBUG_SYNC;
			kl_remove_combe2 << <blocksUV, threads >> >(dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thcombe, thdiff);
			DEBUG_SYNC;
		}
		else {
			cpu_remove_combe2(dstY, srcY, vi.width, vi.height, pitchY, combeY, fpitchY, thcombe, thdiff);
			cpu_remove_combe2(dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thcombe, thdiff);
			cpu_remove_combe2(dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thcombe, thdiff);
		}
	}

	template <typename pixel_t>
	void VisualizeCombe(PVideoFrame& dst, PVideoFrame& combe, int thresh, IScriptEnvironment2* env)
	{
		// 判定結果を表示
		int blue[] = { 73, 230, 111 };

		const uint8_t* combep = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int combePitch = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int dstPitchY = dst->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int dstPitchUV = dst->GetPitch(PLANAR_U) / sizeof(pixel_t);

		// 色を付ける
		for (int y = 0; y < vi.height; ++y) {
			for (int x = 0; x < vi.width; ++x) {
				int score = combep[(x >> 2) + (y >> 2) * combePitch];

				int* color = nullptr;
				if (score >= thresh) {
					color = blue;
				}

				if (color) {
					int offY = x + y * dstPitchY;
					int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
					dstY[offY] = color[0];
					dstU[offUV] = color[1];
					dstV[offUV] = color[2];
				}
			}
		}
	}

	void MakeSwitchFlag(PVideoFrame& flag, PVideoFrame& flagtmp, PVideoFrame& combe, IScriptEnvironment2* env)
	{
		const uint8_t* srcp = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
		uint8_t* flagp = reinterpret_cast<uint8_t*>(flag->GetWritePtr());
		uint8_t* flagtmpp = reinterpret_cast<uint8_t*>(flagtmp->GetWritePtr());
		
		int height = flag->GetHeight();
		int width = flag->GetRowSize();
		int fpitch = flag->GetPitch();
		int cpitch = combe->GetPitch();

		if (IS_CUDA) {
			dim3 threads(16, 8);
			dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
			kl_combe_to_flag << <blocks, threads >> >(
				flagp, width, height, fpitch, srcp, cpitch);
			DEBUG_SYNC;
			kl_sum_box3x3 << <blocks, threads >> >(
				flagtmpp, flagp, width, height, fpitch, 255);
			DEBUG_SYNC;
			kl_sum_box3x3 << <blocks, threads >> >(
				flagp, flagtmpp, width, height, fpitch, 255);
			DEBUG_SYNC;
		}
		else {
			cpu_combe_to_flag(flagp, width, height, fpitch, srcp, cpitch);
			cpu_sum_box3x3(flagtmpp, flagp, width, height, fpitch, 255);
			cpu_sum_box3x3(flagp, flagtmpp, width, height, fpitch, 255);
		}
	}

	PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
	{
		typedef uint8_t pixel_t;

		PVideoFrame src = child->GetFrame(n, env);
		PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
		PVideoFrame dst = env->NewVideoFrame(vi);
		PVideoFrame combe = env->NewVideoFrame(combvi);
		PVideoFrame combetmp = env->NewVideoFrame(combvi);
		PVideoFrame flag = NewSwitchFlagFrame(vi, 32, 2, env);
		PVideoFrame flagtmp = NewSwitchFlagFrame(vi, 32, 2, env);

		CopyFrame<pixel_t>(src, padded, env);
		PadFrame<pixel_t>(padded, env);
		DetectCombe<pixel_t>(padded, combe, env);
		ExtendBlocks(combe, combetmp, env);
		if (detect_uv) {
			MergeUVCoefs(combe, env);
		}
		ApplyUVCoefs(combe, env);
		RemoveCombe<pixel_t>(dst, padded, combe, (int)thsmooth, (int)smooth, env);
		DetectCombe<pixel_t>(dst, combe, env);
		ExtendBlocks(combe, combetmp, env);
		if (detect_uv) {
			MergeUVCoefs(combe, env);
		}
		MakeSwitchFlag(flag, flagtmp, combe, env);
		dst->SetProps(COMBE_FLAG_STR, flag);

		if (!IS_CUDA && show) {
			VisualizeCombe<pixel_t>(dst, combe, (int)thcombe, env);
			return dst;
		}

		return dst;
	}

public:
	KRemoveCombe(PClip clip, float thsmooth, float smooth, bool uv, bool show, float thcombe, IScriptEnvironment* env)
		: KFMFilterBase(clip)
		, padvi(vi)
		, blockvi(vi)
		, thsmooth(thsmooth)
		, smooth(smooth)
		, detect_uv(uv)
		, show(show)
		, thcombe(thcombe)
	{
		if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

		padvi.height += VPAD * 2;

		combvi.width = vi.width / DC_OVERLAP;
		combvi.height = vi.height / DC_OVERLAP;
		combvi.pixel_type = Get8BitType(vi);

		blockvi.width = nblocks(vi.width, OVERLAP);
		blockvi.height = nblocks(vi.height, OVERLAP);
		blockvi.pixel_type = VideoInfo::CS_Y8;
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			return GetFrameT(n, env);
		case 2:
			return GetFrameT(n, env);
		default:
			env->ThrowError("[KRemoveCombe] Unsupported pixel format");
		}

		return PVideoFrame();
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KRemoveCombe(
			args[0].AsClip(),       // source
			(float)args[1].AsFloat(30), // thsmooth
			(float)args[2].AsFloat(100), // smooth
			args[3].AsBool(false), // uv
			args[4].AsBool(false), // show
			(float)args[5].AsFloat(100), // thcombe
			env
		);
	}
};

bool cpu_contains_durty_block(const uint8_t* flagp, int fpitch, int nBlkX, int nBlkY, int* work, int thresh)
{
	for (int by = 0; by < nBlkY; ++by) {
		for (int bx = 0; bx < nBlkX; ++bx) {
			if (flagp[bx + by * fpitch] >= thresh) return true;
		}
	}
	return false;
}

__global__ void kl_init_contains_durty_block(int* work)
{
	*work = 0;
}

__global__ void kl_contains_durty_block(const uint8_t* flagp, int fpitch, int nBlkX, int nBlkY, int* work, int thresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		if (flagp[x + y * fpitch] >= thresh) {
			*work = 1;
		}
	}
}

void cpu_binary_flag(
	uint8_t* dst, int dpitch, const uint8_t* src, int spitch, 
	int nBlkX, int nBlkY, int thresh)
{
	for (int y = 0; y < nBlkY; ++y) {
		for (int x = 0; x < nBlkX; ++x) {
			dst[x + y * dpitch] = ((src[x + y * spitch] >= thresh) ? 128 : 0);
		}
	}
}

__global__ void kl_binary_flag(
	uint8_t* dst, int dpitch, const uint8_t* src, int spitch, 
	int nBlkX, int nBlkY, int thresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		dst[x + y * dpitch] = ((src[x + y * spitch] >= thresh) ? 128 : 0);
	}
}

void cpu_bilinear_x8_v(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int y0 = ((y - 4) >> 3);
			int c0 = ((y0 + 1) << 3) - (y - 4);
			int c1 = 8 - c0;
			auto s0 = src[x + (y0 + 0) * spitch];
			auto s1 = src[x + (y0 + 1) * spitch];
			dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
		}
	}
}

__global__ void kl_bilinear_x8_v(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int y0 = ((y - 4) >> 3);
		int c0 = ((y0 + 1) << 3) - (y - 4);
		int c1 = 8 - c0;
		auto s0 = src[x + (y0 + 0) * spitch];
		auto s1 = src[x + (y0 + 1) * spitch];
		dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
	}
}

void cpu_bilinear_x8_h(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int x0 = ((x - 4) >> 3);
			int c0 = ((x0 + 1) << 3) - (x - 4);
			int c1 = 8 - c0;
			auto s0 = src[(x0 + 0) + y * spitch];
			auto s1 = src[(x0 + 1) + y * spitch];
			dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
		}
	}
}

__global__ void kl_bilinear_x8_h(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int x0 = ((x - 4) >> 3);
		int c0 = ((x0 + 1) << 3) - (x - 4);
		int c1 = 8 - c0;
		auto s0 = src[(x0 + 0) + y * spitch];
		auto s1 = src[(x0 + 1) + y * spitch];
		dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
	}
}

template <typename vpixel_t, typename fpixel_t>
void cpu_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, 
	int width, int height, int pitch, 
	const fpixel_t* flagp, int fpitch,
	int logx, int logy, int nBlkX, int nBlkY)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int4 combe = to_int(flagp[(x << logx) + (y << logy) * fpitch]);
			int4 invcombe = VHelper<int4>::make(128) - combe;
			int4 tmp = (combe * to_int(src60[x + y * pitch]) + invcombe * to_int(src24[x + y * pitch]) + 64) >> 7;
			dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
		}
	}
}

template <typename vpixel_t, typename fpixel_t>
__global__ void kl_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, 
	int width, int height, int pitch,
	const fpixel_t* flagp, int fpitch,
	int logx, int logy, int nBlkX, int nBlkY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int4 combe = to_int(flagp[(x << logx) + (y << logy) * fpitch]);
		int4 invcombe = VHelper<int4>::make(128) - combe;
		int4 tmp = (combe * to_int(src60[x + y * pitch]) + invcombe * to_int(src24[x + y * pitch]) + 64) >> 7;
		dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
	}
}

enum KFMSWTICH_FLAG {
	FRAME_60 = 1,
	FRAME_24,
};

class KFMSwitch : public KFMFilterBase
{
	typedef uint8_t pixel_t;

	PClip clip24;
	PClip fmclip;
	PClip combeclip;
	float thswitch;
	float thpatch;
	bool show;
	bool showflag;

	int logUVx;
	int logUVy;
	int nBlkX, nBlkY;

	VideoInfo workvi;

	PulldownPatterns patterns;

	bool ContainsDurtyBlock(PVideoFrame& flag, PVideoFrame& work, int thpatch, IScriptEnvironment2* env)
	{
		const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());
		int* pwork = reinterpret_cast<int*>(work->GetWritePtr());
		int pitch = flag->GetPitch();

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
			kl_init_contains_durty_block << <1, 1 >> > (pwork);
			kl_contains_durty_block << <blocks, threads >> > (flagp, pitch, nBlkX, nBlkY, pwork, thpatch);
			int result;
			CUDA_CHECK(cudaMemcpy(&result, pwork, sizeof(int), cudaMemcpyDeviceToHost));
			return result != 0;
		}
		else {
			return cpu_contains_durty_block(flagp, pitch, nBlkX, nBlkY, pwork, thpatch);
		}
	}

	void MakeMergeFlag(PVideoFrame& dst, PVideoFrame& src, PVideoFrame& dsttmp, PVideoFrame& srctmp, int thpatch, IScriptEnvironment2* env)
	{
		const uint8_t* srcp = reinterpret_cast<const uint8_t*>(src->GetReadPtr());
		uint8_t* dstp = reinterpret_cast<uint8_t*>(dst->GetWritePtr());
		uint8_t* dsttmpp = reinterpret_cast<uint8_t*>(dsttmp->GetWritePtr()) + dsttmp->GetPitch();
		uint8_t* srctmpp = reinterpret_cast<uint8_t*>(srctmp->GetWritePtr());

		// 0と128の2値にした後、線形補間で画像サイズまで拡大 //

		if (IS_CUDA) {
			dim3 threads(32, 8);
			dim3 binary_blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
			kl_binary_flag << <binary_blocks, threads >> >(
				srctmpp, srctmp->GetPitch(), srcp, src->GetPitch(), nBlkX, nBlkY, thpatch);
			DEBUG_SYNC;
			{
				dim3 threads(32, 1);
				dim3 blocks(nblocks(nBlkX, threads.x));
				kl_padv << <blocks, threads >> > (srctmpp, nBlkX, nBlkY, srctmp->GetPitch(), 1);
				DEBUG_SYNC;
			}
			{
				dim3 threads(1, 32);
				dim3 blocks(1, nblocks(nBlkY, threads.y));
				kl_padh << <blocks, threads >> > (srctmpp, nBlkX, nBlkY + 1 * 2, srctmp->GetPitch(), 1);
				DEBUG_SYNC;
			}
			dim3 h_blocks(nblocks(vi.width, threads.x), nblocks(nBlkY, threads.y));
			kl_bilinear_x8_h << <h_blocks, threads >> >(
				dsttmpp, vi.width, nBlkY + 2, dsttmp->GetPitch(), srctmpp - srctmp->GetPitch(), srctmp->GetPitch());
			DEBUG_SYNC;
			dim3 v_blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
			kl_bilinear_x8_v << <v_blocks, threads >> >(
				dstp, vi.width, vi.height, dst->GetPitch(), dsttmpp + dsttmp->GetPitch(), dsttmp->GetPitch());
			DEBUG_SYNC;
		}
		else {
			cpu_binary_flag(srctmpp, srctmp->GetPitch(), srcp, src->GetPitch(), nBlkX, nBlkY, thpatch);
			cpu_padv(srctmpp, nBlkX, nBlkY, srctmp->GetPitch(), 1);
			cpu_padh(srctmpp, nBlkX, nBlkY + 1 * 2, srctmp->GetPitch(), 1);
			// 上下パディング1行分も含めて処理
			cpu_bilinear_x8_h(dsttmpp, vi.width, nBlkY + 2, dsttmp->GetPitch(), srctmpp - srctmp->GetPitch(), srctmp->GetPitch());
			// ソースはパディング1行分をスキップして渡す
			cpu_bilinear_x8_v(dstp, vi.width, vi.height, dst->GetPitch(), dsttmpp + dsttmp->GetPitch(), dsttmp->GetPitch());
		}
	}

	template <typename pixel_t>
	void MergeBlock(PVideoFrame& src24, PVideoFrame& src60, PVideoFrame& flag, PVideoFrame& dst, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		const vpixel_t* src24Y = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_Y));
		const vpixel_t* src24U = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_U));
		const vpixel_t* src24V = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_V));
		const vpixel_t* src60Y = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_Y));
		const vpixel_t* src60U = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_U));
		const vpixel_t* src60V = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_V));
		vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));
		const uchar4* flagp = reinterpret_cast<const uchar4*>(flag->GetReadPtr());

		int pitchY = src24->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = src24->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;
		int fpitch4 = flag->GetPitch() / sizeof(uchar4);

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_merge << <blocks, threads >> >(
				dstY, src24Y, src60Y, width4, vi.height, pitchY, flagp, fpitch4, 0, 0, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
		}
		else {
			cpu_merge(dstY, src24Y, src60Y, width4, vi.height, pitchY, flagp, fpitch4, 0, 0, nBlkX, nBlkY);
			cpu_merge(dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			cpu_merge(dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
		}
	}

	template <typename pixel_t>
	void VisualizeFlag(PVideoFrame& dst, PVideoFrame& mf, IScriptEnvironment2* env)
	{
		// 判定結果を表示
		int blue[] = { 73, 230, 111 };

		const uint8_t* mfp = reinterpret_cast<const uint8_t*>(mf->GetReadPtr());
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int mfpitch = mf->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int dstPitchY = dst->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int dstPitchUV = dst->GetPitch(PLANAR_U) / sizeof(pixel_t);

		// 色を付ける
		for (int y = 0; y < vi.height; ++y) {
			for (int x = 0; x < vi.width; ++x) {
				int score = mfp[x + y * mfpitch];
				int offY = x + y * dstPitchY;
				int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
				dstY[offY] = (blue[0] * score + dstY[offY] * (128 - score)) >> 7;
				dstU[offUV] = (blue[1] * score + dstU[offUV] * (128 - score)) >> 7;
				dstV[offUV] = (blue[2] * score + dstV[offUV] * (128 - score)) >> 7;
			}
		}
	}

	template <typename pixel_t>
	PVideoFrame InternalGetFrame(int n60, PVideoFrame& fmframe, int& type, IScriptEnvironment2* env)
	{
		int cycleIndex = n60 / 10;
		int kfmPattern = (int)fmframe->GetProps("KFM_Pattern")->GetInt();
		float kfmCost = (float)fmframe->GetProps("KFM_Cost")->GetFloat();

		if (kfmCost > thswitch || PulldownPatterns::Is30p(kfmPattern)) {
			// コストが高いので60pと判断 or 30pの場合
			PVideoFrame frame60 = child->GetFrame(n60, env);
			type = FRAME_60;
			return frame60;
		}

		type = FRAME_24;

		// 24pフレーム番号を取得
		Frame24Info frameInfo = patterns.GetFrame60(kfmPattern, n60);
		int n24 = frameInfo.cycleIndex * 4 + frameInfo.frameIndex;

		if (frameInfo.frameIndex < 0) {
			// 前に空きがあるので前のサイクル
			n24 = frameInfo.cycleIndex * 4 - 1;
		}
		else if (frameInfo.frameIndex >= 4) {
			// 後ろのサイクルのパターンを取得
			PVideoFrame nextfmframe = fmclip->GetFrame(cycleIndex + 1, env);
			int nextPattern = (int)nextfmframe->GetProps("KFM_Pattern")->GetInt();
			int fstart = patterns.GetFrame24(nextPattern, 0).fieldStartIndex;
			if (fstart > 0) {
				// 前に空きがあるので前のサイクル
				n24 = frameInfo.cycleIndex * 4 + 3;
			}
			else {
				// 前に空きがないので後ろのサイクル
				n24 = frameInfo.cycleIndex * 4 + 4;
			}
		}

		PVideoFrame frame24 = clip24->GetFrame(n24, env);
		PVideoFrame flag = combeclip->GetFrame(n24, env)->GetProps(COMBE_FLAG_STR)->GetFrame();

		{
			PVideoFrame work = env->NewVideoFrame(workvi);
			if (ContainsDurtyBlock(flag, work, (int)thpatch, env) == false) {
				// ダメなブロックはないのでそのまま返す
				return frame24;
			}
		}

		PVideoFrame frame60 = child->GetFrame(n60, env);

		VideoInfo mfvi = vi;
		mfvi.pixel_type = VideoInfo::CS_Y8;
		PVideoFrame mflag = env->NewVideoFrame(mfvi);

		{
			// マージ用フラグ作成
			PVideoFrame mflagtmp = env->NewVideoFrame(mfvi);
			PVideoFrame flagtmp = NewSwitchFlagFrame(vi, 32, 2, env);
			MakeMergeFlag(mflag, flag, mflagtmp, flagtmp, (int)thpatch, env);
		}

		if (!IS_CUDA && vi.ComponentSize() == 1 && showflag) {
			env->MakeWritable(&frame24);
			VisualizeFlag<pixel_t>(frame24, mflag, env);
			return frame24;
		}

		// ダメなブロックは60pフレームからコピー
		PVideoFrame dst = env->NewVideoFrame(vi);
		MergeBlock<pixel_t>(frame24, frame60, mflag, dst, env);

		return dst;
	}

	void DrawInfo(PVideoFrame& dst, const char* fps, int pattern, float score, IScriptEnvironment* env) {
		env->MakeWritable(&dst);

		char buf[100]; sprintf(buf, "KFMSwitch: %s pattern:%2d cost:%.1f", fps, pattern, score);
		DrawText(dst, true, 0, 0, buf);
	}

public:
	KFMSwitch(PClip clip60, PClip clip24, PClip fmclip, PClip combeclip,
		float thswitch, float thpatch, bool show, bool showflag, IScriptEnvironment* env)
		: KFMFilterBase(clip60)
		, clip24(clip24)
		, fmclip(fmclip)
		, combeclip(combeclip)
		, thswitch(thswitch)
		, thpatch(thpatch)
		, show(show)
		, showflag(showflag)
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{
		if (vi.width & 7) env->ThrowError("[KFMSwitch]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KFMSwitch]: height must be multiple of 8");

		nBlkX = nblocks(vi.width, OVERLAP);
		nBlkY = nblocks(vi.height, OVERLAP);

		int work_bytes = sizeof(int);
		workvi.pixel_type = VideoInfo::CS_BGR32;
		workvi.width = 4;
		workvi.height = nblocks(work_bytes, workvi.width * 4);
	}

	PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

		int cycleIndex = n60 / 10;
		PVideoFrame fmframe = fmclip->GetFrame(cycleIndex, env);
		int frameType;

		PVideoFrame dst;
		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			dst = InternalGetFrame<uint8_t>(n60, fmframe, frameType, env);
			break;
		case 2:
			dst = InternalGetFrame<uint16_t>(n60, fmframe, frameType, env);
			break;
		default:
			env->ThrowError("[KFMSwitch] Unsupported pixel format");
			break;
		}

		if (!IS_CUDA && pixelSize == 1 && show) {
			const std::pair<int, float>* pfm = (std::pair<int, float>*)fmframe->GetReadPtr();
			const char* fps = (frameType == FRAME_60) ? "60p" : "24p";
			DrawInfo(dst, fps, pfm->first, pfm->second, env);
		}

		return dst;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KFMSwitch(
			args[0].AsClip(),           // clip60
			args[1].AsClip(),           // clip24

			args[2].AsClip(),           // fmclip
			args[3].AsClip(),           // combeclip
			(float)args[4].AsFloat(0.8f),// thswitch
			(float)args[5].AsFloat(40.0f),// thpatch
			args[6].AsBool(false),      // show
			args[7].AsBool(false),      // showflag
			env
			);
	}
};

class AssertOnCUDA : public GenericVideoFilter
{
public:
	AssertOnCUDA(PClip clip) : GenericVideoFilter(clip) { }

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_DEV_TYPE) {
			return DEV_TYPE_CUDA;
		}
		return 0;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new AssertOnCUDA(args[0].AsClip());
	}
};

void AddFuncFMKernel(IScriptEnvironment* env)
{
	env->AddFunction("KTemporalDiff", "c", KTemporalDiff::Create, 0);
	env->AddFunction("KAnalyzeStatic", "c[thcombe]f[thdiff]f", KAnalyzeStatic::Create, 0);
  env->AddFunction("KMergeStatic", "ccc", KMergeStatic::Create, 0);

  env->AddFunction("KFMFrameAnalyzeShow", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyzeShow::Create, 0);
  env->AddFunction("KFMFrameAnalyze", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyze::Create, 0);

  env->AddFunction("KFMFrameAnalyzeCheck", "cc", KFMFrameAnalyzeCheck::Create, 0);

  env->AddFunction("KTelecine", "cc[show]b", KTelecine::Create, 0);
	env->AddFunction("KRemoveCombe", "c[thsmooth]f[smooth]f[uv]b[show]b[thcombe]f", KRemoveCombe::Create, 0);
	env->AddFunction("KRemoveCombeCheck", "cc", KRemoveCombeCheck::Create, 0);

	env->AddFunction("KFMSwitch", "cccc[thswitch]f[thpatch]f[show]b[showflag]b", KFMSwitch::Create, 0);

	env->AddFunction("AssertOnCUDA", "c", AssertOnCUDA::Create, 0);
}
