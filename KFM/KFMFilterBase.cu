
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include "CommonFunctions.h"
#include "KFM.h"
#include "VectorFunctions.cuh"
#include "KFMFilterBase.cuh"


int scaleParam(float thresh, int pixelBits)
{
  return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

int Get8BitType(VideoInfo& vi) {
  if (vi.Is420()) return VideoInfo::CS_YV12;
  else if (vi.Is422()) return VideoInfo::CS_YV16;
  else if (vi.Is444()) return VideoInfo::CS_YV24;
  // Ç±ÇÍà»äOÇÕímÇÁÇÒ
  return VideoInfo::CS_BGR24;
}

PVideoFrame NewSwitchFlagFrame(VideoInfo vi, int hpad, int vpad, PNeoEnv env)
{
  typedef typename VectorType<uint8_t>::type vpixel_t;

  VideoInfo blockpadvi = vi;
  blockpadvi.width = nblocks(vi.width, OVERLAP) + hpad * 2;
  blockpadvi.height = nblocks(vi.height, OVERLAP) + vpad * 2;
  blockpadvi.pixel_type = VideoInfo::CS_Y8;
  PVideoFrame frame = env->NewVideoFrame(blockpadvi);

  // É[Éçèâä˙âª
  vpixel_t* flagp = reinterpret_cast<vpixel_t*>(frame->GetWritePtr());
  int pitch = frame->GetPitch() / sizeof(vpixel_t);
  int width = frame->GetPitch() / sizeof(vpixel_t);
  if (IS_CUDA) {
    dim3 threads(32, 8);
    dim3 blocks(nblocks(width, threads.x), nblocks(blockpadvi.height, threads.y));
    kl_fill<vpixel_t, 0> << <blocks, threads >> >(flagp, width, blockpadvi.height, pitch);
  }
  else {
    cpu_fill<vpixel_t, 0>(flagp, width, blockpadvi.height, pitch);
  }

  return env->SubframePlanar(frame,
    hpad * sizeof(uint8_t) + frame->GetPitch(PLANAR_Y) * vpad,
    frame->GetPitch(PLANAR_Y),
    frame->GetRowSize(PLANAR_Y) - hpad * 2 * sizeof(uint8_t),
    frame->GetHeight(PLANAR_Y) - vpad * 2,
    0, 0, 0);
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

template void cpu_average(uchar4* dst, const uchar4* __restrict__ src0, const uchar4* __restrict__ src1, int width, int height, int pitch);
template void cpu_average(ushort4* dst, const ushort4* __restrict__ src0, const ushort4* __restrict__ src1, int width, int height, int pitch);

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

template __global__ void kl_average(uchar4* dst, const uchar4* __restrict__ src0, const uchar4* __restrict__ src1, int width, int height, int pitch);
template __global__ void kl_average(ushort4* dst, const ushort4* __restrict__ src0, const ushort4* __restrict__ src1, int width, int height, int pitch);

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

template void cpu_padv(uint8_t* dst, int width, int height, int pitch, int vpad);

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

template __global__ void kl_padv(uint8_t* dst, int width, int height, int pitch, int vpad);

template <typename pixel_t>
void cpu_padh(pixel_t* dst, int width, int height, int pitch, int hpad)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < hpad; ++x) {
      dst[(-x - 1) + y * pitch] = dst[(x)+y * pitch];
      dst[(width + x) + y * pitch] = dst[(width - x - 1) + y * pitch];
    }
  }
}

template void cpu_padh(uint8_t* dst, int width, int height, int pitch, int hpad);

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

template __global__ void kl_padh(uint8_t* dst, int width, int height, int pitch, int hpad);

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

template void cpu_copy_border(uint8_t* dst,
  const uint8_t* src, int width, int height, int pitch, int vborder);

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

// srefÇÕbase-1ÉâÉCÉì
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
      // ÉtÉâÉOäiî[
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
    // ÉtÉâÉOäiî[
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

template void cpu_and_coefs(uchar4* dstp, const uchar4* diffp,
  int width, int height, int pitch, float invcombe, float invdiff);
template void cpu_and_coefs(ushort4* dstp, const ushort4* diffp,
  int width, int height, int pitch, float invcombe, float invdiff);

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


template <typename pixel_t>
void KFMFilterBase::CopyFrame(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env)
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

template void KFMFilterBase::CopyFrame<uint8_t>(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env);
template void KFMFilterBase::CopyFrame<uint16_t>(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env);


template <typename pixel_t>
void KFMFilterBase::PadFrame(PVideoFrame& dst, PNeoEnv env)
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

template void KFMFilterBase::PadFrame<uint8_t>(PVideoFrame& dst, PNeoEnv env);
template void KFMFilterBase::PadFrame<uint16_t>(PVideoFrame& dst, PNeoEnv env);

template <typename vpixel_t>
void KFMFilterBase::LaunchAnalyzeFrame(uchar4* dst, int dstPitch,
  const vpixel_t* base, const vpixel_t* sref, const vpixel_t* mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS,
  PNeoEnv env)
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

template void KFMFilterBase::LaunchAnalyzeFrame(uchar4* dst, int dstPitch,
  const uchar4* base, const uchar4* sref, const uchar4* mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS,
  PNeoEnv env);
template void KFMFilterBase::LaunchAnalyzeFrame(uchar4* dst, int dstPitch,
  const ushort4* base, const ushort4* sref, const ushort4* mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS,
  PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::AnalyzeFrame(PVideoFrame& f0, PVideoFrame& f1, PVideoFrame& flag,
  const FrameAnalyzeParam* prmY, const FrameAnalyzeParam* prmC, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;

  int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

  // äeÉvÉåÅ[ÉìÇîªíË
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

template void KFMFilterBase::AnalyzeFrame<uint8_t>(PVideoFrame& f0, PVideoFrame& f1, PVideoFrame& flag,
  const FrameAnalyzeParam* prmY, const FrameAnalyzeParam* prmC, PNeoEnv env);
template void KFMFilterBase::AnalyzeFrame<uint16_t>(PVideoFrame& f0, PVideoFrame& f1, PVideoFrame& flag,
  const FrameAnalyzeParam* prmY, const FrameAnalyzeParam* prmC, PNeoEnv env);

void KFMFilterBase::MergeUVFlags(PVideoFrame& flag, PNeoEnv env)
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
void KFMFilterBase::MergeUVCoefs(PVideoFrame& flag, PNeoEnv env)
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

template void KFMFilterBase::MergeUVCoefs<uint8_t>(PVideoFrame& flag, PNeoEnv env);
template void KFMFilterBase::MergeUVCoefs<uint16_t>(PVideoFrame& flag, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::ApplyUVCoefs(PVideoFrame& flag, PNeoEnv env)
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

template void KFMFilterBase::ApplyUVCoefs<uint8_t>(PVideoFrame& flag, PNeoEnv env);
template void KFMFilterBase::ApplyUVCoefs<uint16_t>(PVideoFrame& flag, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::ExtendCoefs(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
  const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
  vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));

  int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
  int width4 = vi.width >> 2;

  if (IS_CUDA) {
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
    kl_extend_coef << <blocks, threads >> >(
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

template void KFMFilterBase::ExtendCoefs<uint8_t>(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env);
template void KFMFilterBase::ExtendCoefs<uint16_t>(PVideoFrame& src, PVideoFrame& dst, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::CompareFields(PVideoFrame& src, PVideoFrame& flag, PNeoEnv env)
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

template void KFMFilterBase::CompareFields<uint8_t>(PVideoFrame& src, PVideoFrame& flag, PNeoEnv env);
template void KFMFilterBase::CompareFields<uint16_t>(PVideoFrame& src, PVideoFrame& flag, PNeoEnv env);

PVideoFrame KFMFilterBase::OffsetPadFrame(const PVideoFrame& frame, PNeoEnv env)
{
  int vpad = VPAD;
  int vpadUV = VPAD >> logUVy;

  return env->SubframePlanar(frame,
    frame->GetPitch(PLANAR_Y) * vpad, frame->GetPitch(PLANAR_Y), frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y) - vpad * 2,
    frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U));
}

KFMFilterBase::KFMFilterBase(PClip _child)
  : GenericVideoFilter(_child)
  , srcvi(vi)
  , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
  , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
{ }

int __stdcall KFMFilterBase::SetCacheHints(int cachehints, int frame_range) {
  if (cachehints == CACHE_GET_DEV_TYPE) {
    return GetDeviceTypes(child) &
      (DEV_TYPE_CPU | DEV_TYPE_CUDA);
  }
  return 0;
}
