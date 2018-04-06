#include "avisynth.h"

#include "../AvsCUDA.h"

#include <stdint.h>
#include "CommonFunctions.h"
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

template <typename pixel_t>
__global__ void kl_copy(
  pixel_t* dst, int dst_pitch, const pixel_t* __restrict__ src, int src_pitch, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * dst_pitch] = src[x + y * src_pitch];
  }
}


class Align : public GenericVideoFilter
{
  int isRGB;
  int logUVx;
  int logUVy;

  const int* GetPlanes() {
    const int planesYUV[] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[] = { PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
    return isRGB ? planesRGB : planesYUV;
  }

  template <typename pixel_t>
  void Proc(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const int *planes = GetPlanes();

    for (int p = 0; p < 4; ++p) {
      if (src->GetPitch(planes[p]) == 0) continue;
      if (IS_CUDA) {
        const vpixel_t* pSrc = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(planes[p]));
        vpixel_t* pDst = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));
        int srcPitch4 = src->GetPitch(planes[p]) / sizeof(vpixel_t);
        int dstPitch4 = dst->GetPitch(planes[p]) / sizeof(vpixel_t);

        int width4 = vi.width / 4;
        int height = vi.height;

        if (p > 0) {
          width4 >>= logUVx;
          height >>= logUVy;
        }

        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
        kl_copy << <blocks, threads >> >(
          pDst, dstPitch4, pSrc, srcPitch4, width4, height);
        DEBUG_SYNC;
      }
      else {
        const uint8_t* pSrc = src->GetReadPtr(planes[p]);
        uint8_t* pDst = dst->GetWritePtr(planes[p]);
        int srcPitch = src->GetPitch(planes[p]);
        int dstPitch = dst->GetPitch(planes[p]);
        int rowSize = src->GetRowSize(planes[p]);
        int height = src->GetHeight(planes[p]);

        env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowSize, height);
      }
    }
  }

  bool IsAligned(const PVideoFrame& frame)
  {
    const int *planes = GetPlanes();
    for (int p = 0; p < 4; ++p) {
      if (frame->GetPitch(planes[p]) == 0) continue;
      const BYTE* ptr = frame->GetReadPtr(planes[p]);
      int pitch = frame->GetPitch(planes[p]);
      int rowSize = frame->GetRowSize(planes[p]);

      const BYTE* alignedPtr = (const BYTE*)(((uintptr_t)ptr + FRAME_ALIGN - 1) & ~(FRAME_ALIGN - 1));
      int alignedRowSize = (rowSize + FRAME_ALIGN - 1) & ~(FRAME_ALIGN - 1);

      if (alignedPtr != ptr) return false;
      if (alignedRowSize != pitch) return false;
    }
    return true;
  }

public:
  Align(PClip child, IScriptEnvironment* env_)
    : GenericVideoFilter(child)
    , isRGB(vi.IsPlanarRGB() || vi.IsPlanarRGBA())
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    PVideoFrame src = child->GetFrame(n, env);
    if (IsAligned(src)) {
      // Šù‚ÉAlign‚³‚ê‚Ä‚¢‚é
      return src;
    }

    PVideoFrame dst = env->NewVideoFrame(vi);

    // for debug
    if (IsAligned(dst) == false) {
      env->ThrowError("[Align]: New allocated frame is not aligned !!!");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      Proc<uint8_t>(dst, src, env);
      break;
    case 2:
      Proc<uint16_t>(dst, src, env);
      break;
    default:
      env->ThrowError("[Align] Unsupported pixel format");
    }

    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceType(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new Align(
      args[0].AsClip(),
      env);
  }
};

extern const FuncDefinition generic_filters[] = {
  { "Align",  BUILTIN_FUNC_PREFIX,  "c", Align::Create, 0 },
  { 0 }
};
