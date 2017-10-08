
#include <stdint.h>
#include <Windows.h>
#include <avisynth.h>

#include "CommonFunctions.h"
#include "KFM.h"

#include "VectorFunctions.cuh"

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

class CUDAFilterBase : public GenericVideoFilter {
public:
  CUDAFilterBase(PClip _child) : GenericVideoFilter(_child) { }
  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceType(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };
};

template <typename pixel_t>
void cpu_copy(pixel_t* dst, const pixel_t* src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_copy(pixel_t* dst, const pixel_t* src, int width, int height, int pitch)
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

  if (x < width && y < vpad) {
    dst[x + (-y - 1) * pitch] = dst[x + (y)* pitch];
    dst[x + (height + y) * pitch] = dst[x + (height - y - 1)* pitch];
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
__global__ void kl_calc_combe(vpixel_t* dst, const vpixel_t* src, int width, int height, int pitch)
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

template <typename vpixel_t>
void cpu_extend_flag(vpixel_t* dst, const vpixel_t* src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 tmp = max(to_int(src[x + (y - 1) * pitch]), max(to_int(src[x + y * pitch]), to_int(src[x + (y + 1) * pitch])));
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_extend_flag(vpixel_t* dst, const vpixel_t* src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 tmp = max(to_int(src[x + (y - 1) * pitch]), max(to_int(src[x + y * pitch]), to_int(src[x + (y + 1) * pitch])));
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

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

      // ƒtƒ‰ƒOŠi”[
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_compare_frames(vpixel_t* dst,
  const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2,
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

    // ƒtƒ‰ƒOŠi”[
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
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
  const pixel_t* fU, const pixel_t* fV,
  int width, int height, int pitchY, int pitchUV, int logUVx, int logUVy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
    fY[x + y * pitchY] = max(fY[x + y * pitchY], max(fU[offUV], fV[offUV]));
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
  const pixel_t* fY, pixel_t* fU, pixel_t* fV,
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

template <typename vpixel_t>
__global__ void kl_and_coefs(vpixel_t* dstp, const vpixel_t* diffp,
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

class KAnalyzeStatic : public CUDAFilterBase
{
  enum {
    DIST = 1,
    N_REFS = 3,
  };

  typedef uint8_t pixel_t;

  VideoInfo padvi;

  float thcombe;
  float thdiff;

  int logUVx;
  int logUVy;

  PVideoFrame GetRefFrame(int ref, IScriptEnvironment2* env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return child->GetFrame(ref, env);
  }

  void PadFrame(PVideoFrame& dst, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = dst->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = dst->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;
    int vpadUV = VPAD >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, VPAD);
      dim3 blocks(nblocks(width4, threads.x));
      dim3 threadsUV(32, vpadUV);
      dim3 blocksUV(nblocks(width4UV, threads.x));
      kl_padv << <blocks, threads >> >(dstY, width4, vi.height, pitchY, VPAD);
      DEBUG_SYNC;
      kl_padv << <blocksUV, threadsUV >> >(dstU, width4UV, heightUV, pitchUV, vpadUV);
      DEBUG_SYNC;
      kl_padv << <blocksUV, threadsUV >> >(dstV, width4UV, heightUV, pitchUV, vpadUV);
      DEBUG_SYNC;
    }
    else {
      cpu_padv<vpixel_t>(dstY, width4, vi.height, pitchY, VPAD);
      cpu_padv<vpixel_t>(dstU, width4UV, heightUV, pitchUV, vpadUV);
      cpu_padv<vpixel_t>(dstV, width4UV, heightUV, pitchUV, vpadUV);
    }
  }

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
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_copy<<<blocks, threads>>>(dstY, srcY, width4, vi.height, pitchY);
      DEBUG_SYNC;
      kl_copy<<<blocksUV, threads>>>(dstU, srcU, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_copy<<<blocksUV, threads>>>(dstV, srcV, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_copy<vpixel_t>(dstY, srcY, width4, vi.height, pitchY);
      cpu_copy<vpixel_t>(dstU, srcU, width4UV, heightUV, pitchUV);
      cpu_copy<vpixel_t>(dstV, srcV, width4UV, heightUV, pitchUV);
    }
  }

  void CompareFields(PVideoFrame& src, PVideoFrame& flag, int thresh, IScriptEnvironment2* env)
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

  void ExtendFlag(pixel_t* dst, const pixel_t* src, int width, int height, int pitch)
  {
    for (int y = 1; y < height - 1; ++y) {
      for (int x = 0; x < width; ++x) {
        dst[x + y * pitch] = max(src[x + (y - 1) * pitch], max(src[x + y * pitch], src[x + (y + 1) * pitch]));
      }
    }
  }

  void ExtendFlag(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment2* env)
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
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_extend_flag << <blocks, threads >> >(
        dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
      DEBUG_SYNC;
      kl_extend_flag << <blocksUV, threads >> >(
        dstU + pitchUV, srcU + pitchUV, width4UV, heightUV - 2, pitchUV);
      DEBUG_SYNC;
      kl_extend_flag << <blocksUV, threads >> >(
        dstV + pitchUV, srcV + pitchUV, width4UV, heightUV - 2, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_extend_flag(dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
      cpu_extend_flag(dstU + pitchUV, srcU + pitchUV, width4UV, heightUV - 2, pitchUV);
      cpu_extend_flag(dstV + pitchUV, srcV + pitchUV, width4UV, heightUV - 2, pitchUV);
    }
  }

  void CompareFrames(PVideoFrame* frames, PVideoFrame& flag, int thdiff, IScriptEnvironment2* env)
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
        fU, fV, vi.width, vi.height, pitchY, pitchUV,  logUVx, logUVy);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_uvcoefs(fY,
        fU, fV, vi.width, vi.height, pitchY, pitchUV, logUVx, logUVy);
    }
  }

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

  void AndCoefs(PVideoFrame& dst, PVideoFrame& flagd, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* diffp = reinterpret_cast<const vpixel_t*>(flagd->GetReadPtr());
    vpixel_t* dstp = reinterpret_cast<vpixel_t*>(dst->GetWritePtr());
    int pitch = dst->GetPitch() / sizeof(vpixel_t);

    // dst: combe‚ ‚èƒtƒ‰ƒO
    // flagd: diff‚ ‚èƒtƒ‰ƒO
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

  PVideoFrame OffsetPadFrame(PVideoFrame& frame, IScriptEnvironment2* env)
  {
    int vpad = VPAD;
    int vpadUV = VPAD >> logUVy;

    return env->SubframePlanar(frame,
      frame->GetPitch(PLANAR_Y) * vpad, frame->GetPitch(PLANAR_Y), frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y) - vpad * 2,
      frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U));
  }

public:
  KAnalyzeStatic(PClip clip30, float thcombe, float thdiff, IScriptEnvironment2* env)
    : CUDAFilterBase(clip30)
    , thcombe(thcombe)
    , thdiff(thdiff)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , padvi(vi)
  {
    if (logUVx != 1 || logUVy != 1) env->ThrowError("[KAnalyzeStatic] Unsupported format (only supports YV12)");

    padvi.height += VPAD * 2;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    PVideoFrame frames[N_REFS];
    for (int i = 0; i < N_REFS; ++i) {
      frames[i] = GetRefFrame(i + n - DIST, env);
    }
    PVideoFrame& src = frames[DIST];
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame flagtmp = env->NewVideoFrame(vi);
    PVideoFrame flagc = env->NewVideoFrame(vi);
    PVideoFrame flagd = env->NewVideoFrame(vi);

    CopyFrame(src, padded, env);
    PadFrame(padded, env);
    CompareFields(padded, flagtmp, (int)thcombe, env);
    MergeUVCoefs(flagtmp, env);
    ExtendFlag(flagtmp, flagc, env);

    CompareFrames(frames, flagtmp, (int)thdiff, env);
    MergeUVCoefs(flagtmp, env);
    ExtendFlag(flagtmp, flagd, env);

    AndCoefs(flagc, flagd, env); // combe‚ ‚èdiff‚È‚µ -> flagc
    ApplyUVCoefs(flagc, env);

    return flagc;
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

void AddFuncFMKernel(IScriptEnvironment* env)
{
  env->AddFunction("KAnalyzeStatic", "c[thcombe]f[thdiff]f", KAnalyzeStatic::Create, 0);
}
