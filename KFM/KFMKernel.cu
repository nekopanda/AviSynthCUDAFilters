
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
  return (a + c * 4 + e - (b + d) * 3);
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
void cpu_compare_frames(vpixel_t* dst,
  const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2,
  int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 a = to_int(src0[x + y * pitch]);
      int4 b = to_int(src1[x + y * pitch]);
      int4 c = to_int(src2[x + y * pitch]);

      int4 minv = min(a, min(b, c));
      int4 maxv = max(a, max(b, c));

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
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 a = to_int(src0[x + y * pitch]);
    int4 b = to_int(src1[x + y * pitch]);
    int4 c = to_int(src2[x + y * pitch]);

    int4 minv = min(a, min(b, c));
    int4 maxv = max(a, max(b, c));

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

class KAnalyzeStatic : public KFMFilterBase
{
  enum {
    DIST = 1,
    N_REFS = 3,
  };

  VideoInfo padvi;

  float thcombe;
  float thdiff;

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
        srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
      DEBUG_SYNC;
      kl_compare_frames << <blocksUV, threads >> >(dstU,
        srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_compare_frames << <blocksUV, threads >> >(dstV,
        srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_compare_frames(dstY, srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
      cpu_compare_frames(dstU, srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
      cpu_compare_frames(dstV, srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
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
    PVideoFrame frames[N_REFS];
    for (int i = 0; i < N_REFS; ++i) {
      frames[i] = GetRefFrame(i + n - DIST, env);
    }
    PVideoFrame& src = frames[DIST];
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame flagtmp = env->NewVideoFrame(vi);
    PVideoFrame flagc = env->NewVideoFrame(vi);
    PVideoFrame flagd = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(src, padded, env);
    PadFrame<pixel_t>(padded, env);
    CompareFields<pixel_t>(padded, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagc, env);

    CompareFrames<pixel_t>(frames, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagd, env);

    AndCoefs<pixel_t>(flagc, flagd, env); // combeありdiffなし -> flagc
    ApplyUVCoefs<pixel_t>(flagc, env);

    return flagc;
  }

public:
  KAnalyzeStatic(PClip clip30, float thcombe, float thdiff, IScriptEnvironment2* env)
    : KFMFilterBase(clip30)
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
    return new KAnalyzeStatic(
      args[0].AsClip(),       // clip30
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
      args[1].AsInt(15),       // threshMY
      args[2].AsInt(7),       // threshSY
      args[3].AsInt(20),       // threshMC
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
  void CreateWeaveFrame2F(const PVideoFrame& srct, const PVideoFrame& srcb, const PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srctY = reinterpret_cast<const vpixel_t*>(srct->GetReadPtr(PLANAR_Y));
    const vpixel_t* srctU = reinterpret_cast<const vpixel_t*>(srct->GetReadPtr(PLANAR_U));
    const vpixel_t* srctV = reinterpret_cast<const vpixel_t*>(srct->GetReadPtr(PLANAR_V));
    const vpixel_t* srcbY = reinterpret_cast<const vpixel_t*>(srcb->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcbU = reinterpret_cast<const vpixel_t*>(srcb->GetReadPtr(PLANAR_U));
    const vpixel_t* srcbV = reinterpret_cast<const vpixel_t*>(srcb->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = srct->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = srct->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
      // copy top
      kl_copy << <blocks, threads >> >(dstY, srctY, width4, vi.height / 2, pitchY * 2);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstU, srctU, width4UV, heightUV / 2, pitchUV * 2);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstV, srctV, width4UV, heightUV / 2, pitchUV * 2);
      DEBUG_SYNC;
      // copy bottom
      kl_copy << <blocks, threads >> >(dstY + pitchY, srcbY + pitchY, width4, vi.height / 2, pitchY * 2);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstU + pitchUV, srcbU + pitchUV, width4UV, heightUV / 2, pitchUV * 2);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads >> >(dstV + pitchUV, srcbV + pitchUV, width4UV, heightUV / 2, pitchUV * 2);
      DEBUG_SYNC;
    }
    else {
      // copy top
      cpu_copy(dstY, srctY, width4, vi.height / 2, pitchY * 2);
      cpu_copy(dstU, srctU, width4UV, heightUV / 2, pitchUV * 2);
      cpu_copy(dstV, srctV, width4UV, heightUV / 2, pitchUV * 2);
      // copy bottom
      cpu_copy(dstY + pitchY, srcbY + pitchY, width4, vi.height / 2, pitchY * 2);
      cpu_copy(dstU + pitchUV, srcbU + pitchUV, width4UV, heightUV / 2, pitchUV * 2);
      cpu_copy(dstV + pitchUV, srcbV + pitchUV, width4UV, heightUV / 2, pitchUV * 2);
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
      if (parity) {
        CreateWeaveFrame2F<pixel_t>(nxt, cur, dst, env);
      }
      else {
        CreateWeaveFrame2F<pixel_t>(cur, nxt, dst, env);
      }
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
void cpu_remove_combe(pixel_t* dst,
  const pixel_t* src, const pixel_t* combe,
  int width, int height, int pitch, int thcombe, int thdiff)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (combe[x + y * pitch] < thcombe) {
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
__global__ void kl_remove_combe(pixel_t* dst,
  const pixel_t* src, const pixel_t* combe,
  int width, int height, int pitch, int thcombe, int thdiff)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    if (combe[x + y * pitch] < thcombe) {
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

enum {
  RC_COUNT_TH_W = BLOCK_SIZE / 4,
  RC_COUNT_TH_H = BLOCK_SIZE,
  RC_COUNT_THREADS = RC_COUNT_TH_W * RC_COUNT_TH_H,
};

template <typename vpixel_t>
void cpu_count_cmflags(uint8_t* flagp,
  const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int thcombe, int ratio1, int ratio2)
{
  for (int by = 0; by < nBlkY - 1; ++by) {
    for (int bx = 0; bx < nBlkX - 1; ++bx) {
      int yStart = by * OVERLAP;
      int yEnd = yStart + BLOCK_SIZE;
      int xStart = bx * OVERLAP / 4;
      int xEnd = xStart + BLOCK_SIZE / 4;

      int sum = 0;
      for (int y = yStart; y < yEnd; ++y) {
        for (int x = xStart; x < xEnd; ++x) {
          vpixel_t srcv = srcp[x + y * pitch];

          // 横4ピクセルは1個とみなす
          int b =
            (srcv.x >= thcombe) |
            (srcv.y >= thcombe) |
            (srcv.z >= thcombe) |
            (srcv.w >= thcombe);

          sum += b ? 1 : 0;
        }
      }

      uint8_t flag = (sum > ratio1) | ((sum > ratio2) << 1);
      flagp[(bx + 1) + (by + 1) * nBlkX] = flag;
    }
  }
}

template <typename vpixel_t>
__global__ void kl_count_cmflags(uint8_t* flagp,
  const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int thcombe, int ratio1, int ratio2)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = tx + ty * RC_COUNT_TH_W;

  vpixel_t srcv = srcp[(bx * OVERLAP / 4 + tx) + (by * OVERLAP + ty) * pitch];

  // 横4ピクセルは1個とみなす
  int sum = 
    (srcv.x >= thcombe) |
    (srcv.y >= thcombe) |
    (srcv.z >= thcombe) |
    (srcv.w >= thcombe);

  __shared__ int sbuf[RC_COUNT_THREADS];
  dev_reduce<int, RC_COUNT_THREADS, AddReducer<int>>(tid, sum, sbuf);

  if (tid == 0) {
    uint8_t flag = (sum > ratio1) | ((sum > ratio2) << 1);
    flagp[(bx + 1) + (by + 1) * nBlkX] = flag;
  }
}

void cpu_clean_blocks(uint8_t* dstp, const uint8_t* srcp, int nBlkX, int nBlkY)
{
  // 書き込む予定がないところをゼロにする
  for (int bx = 0; bx < nBlkX; ++bx) {
    dstp[bx] = 0;
  }
  for (int by = 1; by < nBlkY; ++by) {
    dstp[by * nBlkX] = 0;
  }

  for (int by = 1; by < nBlkY; ++by) {
    for (int bx = 1; bx < nBlkX; ++bx) {
      dstp[bx + by * nBlkX] = srcp[bx + by * nBlkX];

      if (srcp[bx + by * nBlkX] == 1) {

        int yStart = std::max(by - 2, 1);
        int yEnd = std::min(by + 2, nBlkY - 1);
        int xStart = std::max(bx - 2, 1);
        int xEnd = std::min(bx + 2, nBlkX - 1);

        bool isOK = true;
        for (int y = yStart; y <= yEnd; ++y) {
          for (int x = xStart; x <= xEnd; ++x) {
            if (srcp[x + y * nBlkX] & 2) {
              // デカイ縞
              isOK = false;
            }
          }
        }

        if (isOK) {
          dstp[bx + by * nBlkX] = 0;
        }
      }
    }
  }
}

__global__ void kl_clean_blocks(uint8_t* dstp, const uint8_t* srcp, int nBlkX, int nBlkY)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;

  if (bx < nBlkX && by < nBlkY) {
    if (bx == 0 || by == 0) {
      // 書き込む予定がないところをゼロにする
      dstp[bx + by * nBlkX] = 0;
    }
    else {
      dstp[bx + by * nBlkX] = srcp[bx + by * nBlkX];

      if (srcp[bx + by * nBlkX] == 1) {
        int yStart = max(by - 2, 1);
        int yEnd = min(by + 2, nBlkY - 1);
        int xStart = max(bx - 2, 1);
        int xEnd = min(bx + 2, nBlkX - 1);

        bool isOK = true;
        for (int y = yStart; y <= yEnd; ++y) {
          for (int x = xStart; x <= xEnd; ++x) {
            if (srcp[x + y * nBlkX] & 2) {
              // デカイ縞
              isOK = false;
            }
          }
        }

        if (isOK) {
          dstp[bx + by * nBlkX] = 0;
        }
      }
    }
  }
}

void cpu_extend_blocks(uint8_t* dstp, int nBlkX, int nBlkY)
{
  for (int by = 1; by < nBlkY; ++by) {
    for (int bx = 0; bx < nBlkX - 1; ++bx) {
      dstp[bx + by * nBlkX] |= dstp[bx + 1 + (by + 0) * nBlkX];
    }
  }
  for (int by = 0; by < nBlkY - 1; ++by) {
    for (int bx = 0; bx < nBlkX; ++bx) {
      dstp[bx + by * nBlkX] |= dstp[bx + 0 + (by + 1) * nBlkX];
    }
  }
}

__global__ void kl_extend_blocks_h(uint8_t* dstp, const uint8_t* srcp, int nBlkX, int nBlkY)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;

  if (bx < nBlkX && by < nBlkY) {
    if (bx == nBlkX - 1) {
      // 書き込む予定がないところにソースをコピーする
      dstp[bx + by * nBlkX] = srcp[bx + by * nBlkX];
    }
    else {
      dstp[bx + by * nBlkX] =
        srcp[bx + 0 + (by + 0) * nBlkX] |
        srcp[bx + 1 + (by + 0) * nBlkX];
    }
  }
}

__global__ void kl_extend_blocks_v(uint8_t* dstp, const uint8_t* srcp, int nBlkX, int nBlkY)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;

  if (bx < nBlkX && by < nBlkY) {
    if (by == nBlkY - 1) {
      // 書き込む予定がないところにソースをコピーする
      dstp[bx + by * nBlkX] = srcp[bx + by * nBlkX];
    }
    else {
      dstp[bx + by * nBlkX] =
        srcp[bx + 0 + (by + 0) * nBlkX] |
        srcp[bx + 0 + (by + 1) * nBlkX];
    }
  }
}

class KRemoveCombe : public KFMFilterBase
{
  VideoInfo padvi;
  VideoInfo blockvi;
  int nBlkX, nBlkY;

  float thsmooth;
  float smooth;
  float thcombe;
  float ratio1;
  float ratio2;
  bool outcombe;
  bool show;

  template <typename pixel_t>
  void RemoveCombe(PVideoFrame& dst, PVideoFrame& src, PVideoFrame& flag, int thcombe, int thdiff, IScriptEnvironment2* env)
  {
    const pixel_t* flagY = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_Y));
    const pixel_t* flagU = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_U));
    const pixel_t* flagV = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_V));
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_remove_combe << <blocks, threads >> >(dstY, srcY, flagY, vi.width, vi.height, pitchY, thcombe, thdiff);
      DEBUG_SYNC;
      kl_remove_combe << <blocksUV, threads >> >(dstU, srcU, flagU, widthUV, heightUV, pitchUV, thcombe, thdiff);
      DEBUG_SYNC;
      kl_remove_combe << <blocksUV, threads >> >(dstV, srcV, flagV, widthUV, heightUV, pitchUV, thcombe, thdiff);
      DEBUG_SYNC;
    }
    else {
      cpu_remove_combe(dstY, srcY, flagY, vi.width, vi.height, pitchY, thcombe, thdiff);
      cpu_remove_combe(dstU, srcU, flagU, widthUV, heightUV, pitchUV, thcombe, thdiff);
      cpu_remove_combe(dstV, srcV, flagV, widthUV, heightUV, pitchUV, thcombe, thdiff);
    }
  }

  template <typename pixel_t>
  void VisualizeFlags(PVideoFrame& dst, PVideoFrame& fflag, IScriptEnvironment2* env)
  {
    // 判定結果を表示
    int blue[] = { 73, 230, 111 };

    const pixel_t* fflagp = reinterpret_cast<const pixel_t*>(fflag->GetReadPtr(PLANAR_Y));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int flagPitch = fflag->GetPitch(PLANAR_Y);
    int dstPitchY = dst->GetPitch(PLANAR_Y);
    int dstPitchUV = dst->GetPitch(PLANAR_U);

    // 色を付ける
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int flag = fflagp[x + y * flagPitch];

        int* color = nullptr;
        if (flag) {
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

  template <typename pixel_t>
  void CountFlags(PVideoFrame& block, PVideoFrame& flag, int thcombe, int ratio1, int ratio2, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcp = reinterpret_cast<const vpixel_t*>(flag->GetReadPtr(PLANAR_Y));
    uint8_t* flagp = reinterpret_cast<uint8_t*>(block->GetWritePtr());
    int pitch = flag->GetPitch(PLANAR_Y) / sizeof(vpixel_t);

    if (IS_CUDA) {
      dim3 threads(RC_COUNT_TH_W, RC_COUNT_TH_H);
      dim3 blocks(nBlkX - 1, nBlkY - 1);
      kl_count_cmflags << <blocks, threads >> >(
        flagp, srcp, pitch, nBlkX, nBlkY, thcombe, ratio1, ratio2);
      DEBUG_SYNC;
    }
    else {
      cpu_count_cmflags(flagp, srcp, pitch, nBlkX, nBlkY, thcombe, ratio1, ratio2);
    }
  }

  void CleanBlocks(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment2* env)
  {
    // 周囲にデカイ縞がなければOKとみなす
    uint8_t* srcp = reinterpret_cast<uint8_t*>(src->GetWritePtr());
    uint8_t* dstp = reinterpret_cast<uint8_t*>(dst->GetWritePtr());

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
      kl_clean_blocks << <blocks, threads >> >(dstp, srcp, nBlkX, nBlkY);
      DEBUG_SYNC;
    }
    else {
      cpu_clean_blocks(dstp, srcp, nBlkX, nBlkY);
    }
  }

  void ExtendBlocks(PVideoFrame& dst, PVideoFrame& tmp, IScriptEnvironment2* env)
  {
    uint8_t* tmpp = reinterpret_cast<uint8_t*>(tmp->GetWritePtr());
    uint8_t* dstp = reinterpret_cast<uint8_t*>(dst->GetWritePtr());

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
      kl_extend_blocks_h << <blocks, threads >> >(tmpp, dstp, nBlkX, nBlkY);
      kl_extend_blocks_v << <blocks, threads >> >(dstp, tmpp, nBlkX, nBlkY);
      DEBUG_SYNC;
    }
    else {
      cpu_extend_blocks(dstp, nBlkX, nBlkY);
    }
  }

  PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
  {
    typedef uint8_t pixel_t;

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame dst = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame flag = env->NewVideoFrame(vi);
    PVideoFrame flage = env->NewVideoFrame(vi);
    PVideoFrame blocks = env->NewVideoFrame(blockvi);
    PVideoFrame blockse = env->NewVideoFrame(blockvi);

    CopyFrame<pixel_t>(src, padded, env);
    PadFrame<pixel_t>(padded, env);
    CompareFields<pixel_t>(padded, flag, env);
    MergeUVCoefs<pixel_t>(flag, env);
    ExtendCoefs<pixel_t>(flag, flage, env);
    ApplyUVCoefs<pixel_t>(flage, env);
    RemoveCombe<pixel_t>(dst, padded, flage, (int)thsmooth, (int)smooth, env);
    PadFrame<pixel_t>(dst, env);
    CompareFields<pixel_t>(dst, flag, env);
    MergeUVCoefs<pixel_t>(flag, env);
    CountFlags<pixel_t>(blocks, flag, (int)thcombe, (int)ratio1, (int)ratio2, env);
    CleanBlocks(blockse, blocks, env);
    ExtendBlocks(blockse, blocks, env);

    if (!IS_CUDA && show) {
      VisualizeFlags<pixel_t>(padded, flag, env);
      padded->SetProps(COMBE_FLAG_STR, blockse);
      return padded;
    }

    dst->SetProps(COMBE_FLAG_STR, blockse);
    return dst;
  }

public:
  KRemoveCombe(PClip clip, float thsmooth, float smooth, float thcombe, float ratio1, float ratio2, bool show, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , padvi(vi)
    , thsmooth(thsmooth)
    , smooth(smooth)
    , thcombe(thcombe)
    , ratio1(ratio1)
    , ratio2(ratio2)
    , show(show)
  {
    if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

    padvi.height += VPAD * 2;

    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);

    int flag_bytes = sizeof(uint8_t) * nBlkX * nBlkY;
    blockvi.pixel_type = VideoInfo::CS_BGR32;
    blockvi.width = 2048;
    blockvi.height = nblocks(flag_bytes, blockvi.width * 4);
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
      (float)args[2].AsFloat(50), // smooth
      (float)args[3].AsFloat(150), // thcombe
      (float)args[4].AsFloat(0), // ratio1
      (float)args[5].AsFloat(5), // ratio2
      args[6].AsBool(false), // show
      env
    );
  }
};

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

bool cpu_contains_durty_block(const uint8_t* flagp, int nBlkX, int nBlkY, int* work)
{
	for (int by = 0; by < nBlkY; ++by) {
		for (int bx = 0; bx < nBlkX; ++bx) {
			if (flagp[bx + by * nBlkX]) return true;
		}
	}
	return false;
}

__global__ void kl_init_contains_durty_block(int* work)
{
	*work = 0;
}

__global__ void kl_contains_durty_block(const uint8_t* flagp, int nBlkX, int nBlkY, int* work)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		if (flagp[x + y * nBlkX]) {
			*work = 1;
		}
	}
}

template <typename vpixel_t>
void cpu_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, const uint8_t* flagp,
	int width, int height, int pitch, int bshiftX, int bshiftY, int nBlkX, int nBlkY)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			bool isCombe = flagp[(x >> bshiftX) + (y >> bshiftY) * nBlkX] != 0;
			dst[x + y * pitch] = (isCombe ? src60 : src24)[x + y * pitch];
		}
	}
}

template <typename vpixel_t>
__global__ void kl_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, const uint8_t* flagp,
	int width, int height, int pitch, int bshiftX, int bshiftY, int nBlkX, int nBlkY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		bool isCombe = flagp[(x >> bshiftX) + (y >> bshiftY) * nBlkX] != 0;
		dst[x + y * pitch] = (isCombe ? src60 : src24)[x + y * pitch];
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
	float thresh;
	bool show;

	int logUVx;
	int logUVy;
	int nBlkX, nBlkY;

	VideoInfo workvi;

	PulldownPatterns patterns;

	bool ContainsDurtyBlock(PVideoFrame& flag, PVideoFrame& work, IScriptEnvironment2* env)
	{
		const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());
		int* pwork = reinterpret_cast<int*>(work->GetWritePtr());

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
			kl_init_contains_durty_block << <1, 1 >> > (pwork);
			kl_contains_durty_block << <blocks, threads >> > (flagp, nBlkX, nBlkY, pwork);
			int result;
			CUDA_CHECK(cudaMemcpy(&result, pwork, sizeof(int), cudaMemcpyDeviceToHost));
			return result != 0;
		}
		else {
			return cpu_contains_durty_block(flagp, nBlkX, nBlkY, pwork);
		}
	}

	template <typename pixel_t>
	void MergeBlock(PVideoFrame& src24, PVideoFrame& src60, PVideoFrame& flag, PVideoFrame& dst, IScriptEnvironment2* env)
	{
		const pixel_t* src24Y = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_Y));
		const pixel_t* src24U = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_U));
		const pixel_t* src24V = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_V));
		const pixel_t* src60Y = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_Y));
		const pixel_t* src60U = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_U));
		const pixel_t* src60V = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));
		const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());

		int pitchY = src24->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int pitchUV = src24->GetPitch(PLANAR_U) / sizeof(pixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;
		int bshiftUVx = 1 - logUVx;
		int bshiftUVy = 3 - logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_merge << <blocks, threads >> >(
				dstY, src24Y, src60Y, flagp, width4, vi.height, pitchY, 1, 3, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstU, src24U, src60U, flagp, width4UV, heightUV, pitchUV, bshiftUVx, bshiftUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstV, src24V, src60U, flagp, width4UV, heightUV, pitchUV, bshiftUVx, bshiftUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
		}
		else {
			cpu_merge(dstY, src24Y, src60Y, flagp, width4, vi.height, pitchY, 1, 3, nBlkX, nBlkY);
			cpu_merge(dstU, src24U, src60U, flagp, width4UV, heightUV, pitchUV, bshiftUVx, bshiftUVy, nBlkX, nBlkY);
			cpu_merge(dstV, src24V, src60U, flagp, width4UV, heightUV, pitchUV, bshiftUVx, bshiftUVy, nBlkX, nBlkY);
		}
	}

	template <typename pixel_t>
	PVideoFrame InternalGetFrame(int n60, PVideoFrame& fmframe, int& type, IScriptEnvironment2* env)
	{
		int cycleIndex = n60 / 10;
		int kfmPattern = (int)fmframe->GetProps("KFM_Pattern")->GetInt();
		float kfmCost = (float)fmframe->GetProps("KFM_Cost")->GetFloat();

		if (kfmCost > thresh) {
			// コストが高いので60pと判断
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
			const std::pair<int, float>* pnextfm = (std::pair<int, float>*)nextfmframe->GetReadPtr();
			int fstart = patterns.GetFrame24(pnextfm->first, 0).fieldStartIndex;
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

		PVideoFrame work = env->NewVideoFrame(workvi);
		if (ContainsDurtyBlock(flag, work, env) == false) {
			// ダメなブロックはないのでそのまま返す
			return frame24;
		}

		// ダメなブロックは60pフレームからコピー
		PVideoFrame frame60 = child->GetFrame(n60, env);
		PVideoFrame dst = env->NewVideoFrame(vi);

		MergeBlock<pixel_t>(frame24, frame60, flag, dst, env);

		return dst;
	}

	void DrawInfo(PVideoFrame& dst, const char* fps, int pattern, float score, IScriptEnvironment* env) {
		env->MakeWritable(&dst);

		char buf[100]; sprintf(buf, "KFMSwitch: %s pattern:%2d cost:%.1f", fps, pattern, score);
		DrawText(dst, true, 0, 0, buf);
	}

public:
	KFMSwitch(PClip clip60, PClip clip24, PClip fmclip, PClip combeclip, float thresh, bool show, IScriptEnvironment* env)
		: KFMFilterBase(clip60)
		, clip24(clip24)
		, fmclip(fmclip)
		, combeclip(combeclip)
		, thresh(thresh)
		, show(show)
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
			(float)args[4].AsFloat(),   // thresh
			args[5].AsBool(),           // show
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
  env->AddFunction("KAnalyzeStatic", "c[thcombe]f[thdiff]f", KAnalyzeStatic::Create, 0);
  env->AddFunction("KMergeStatic", "ccc", KMergeStatic::Create, 0);

  env->AddFunction("KFMFrameAnalyzeShow", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyzeShow::Create, 0);
  env->AddFunction("KFMFrameAnalyze", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyze::Create, 0);

  env->AddFunction("KFMFrameAnalyzeCheck", "cc", KFMFrameAnalyzeCheck::Create, 0);

  env->AddFunction("KTelecine", "cc[show]b", KTelecine::Create, 0);
  env->AddFunction("KRemoveCombe", "c[thsmooth]f[smooth]f[thcombe]f[ratio1]f[ratio2]f[show]b", KRemoveCombe::Create, 0);
	env->AddFunction("KRemoveCombeCheck", "cc", KRemoveCombeCheck::Create, 0);

	env->AddFunction("KFMSwitch", "cccc[thresh]f[show]b", KFMSwitch::Create, 0);

	env->AddFunction("AssertOnCUDA", "c", AssertOnCUDA::Create, 0);
}
