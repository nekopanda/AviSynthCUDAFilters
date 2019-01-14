
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include "CommonFunctions.h"
#include "KFM.h"
#include "Copy.h"
#include "VectorFunctions.cuh"
#include "KFMFilterBase.cuh"


int scaleParam(float thresh, int pixelBits)
{
  return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

int Get8BitType(const VideoInfo& vi) {
  if (vi.Is420()) return VideoInfo::CS_YV12;
  else if (vi.Is422()) return VideoInfo::CS_YV16;
  else if (vi.Is444()) return VideoInfo::CS_YV24;
  // これ以外は知らん
  return VideoInfo::CS_BGR24;
}

int Get16BitType(const VideoInfo& vi) {
  if (vi.Is420()) return VideoInfo::CS_YUV420P16;
  else if (vi.Is422()) return VideoInfo::CS_YUV422P16;
  else if (vi.Is444()) return VideoInfo::CS_YUV444P16;
  // これ以外は知らん
  return VideoInfo::CS_BGR48;
}

int GetYType(const VideoInfo& vi) {
	switch (vi.BitsPerComponent()) {
	case 8: return VideoInfo::CS_Y8;
	case 10: return VideoInfo::CS_Y10;
	case 12: return VideoInfo::CS_Y12;
	case 14: return VideoInfo::CS_Y14;
	case 16: return VideoInfo::CS_Y16;
	case 32: return VideoInfo::CS_Y32;
	}
	// これ以外は知らん
	return VideoInfo::CS_Y8;
}

int Get444Type(const VideoInfo& vi) {
	switch (vi.BitsPerComponent()) {
	case 8: return VideoInfo::CS_YV24;
	case 10: return VideoInfo::CS_YUV444P10;
	case 12: return VideoInfo::CS_YUV444P12;
	case 14: return VideoInfo::CS_YUV444P14;
	case 16: return VideoInfo::CS_YUV444P16;
	case 32: return VideoInfo::CS_YUV444PS;
	}
	// これ以外は知らん
	return VideoInfo::CS_YV24;
}

Frame NewSwitchFlagFrame(VideoInfo vi, PNeoEnv env)
{
  typedef typename VectorType<uint8_t>::type vpixel_t;
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

  Frame frame = env->NewVideoFrame(vi);

  // ゼロ初期化
  vpixel_t* flagp = frame.GetWritePtr<vpixel_t>();
  int pitch = frame.GetPitch<vpixel_t>();
  int width = frame.GetPitch<vpixel_t>();
  if (IS_CUDA) {
    dim3 threads(32, 8);
    dim3 blocks(nblocks(width, threads.x), nblocks(vi.height, threads.y));
    kl_fill<vpixel_t, 0> << <blocks, threads, 0, stream >> > (flagp, width, vi.height, pitch);
  }
  else {
    cpu_fill<vpixel_t, 0>(flagp, width, vi.height, pitch);
  }

  frame.Crop(COMBE_FLAG_PAD_W, COMBE_FLAG_PAD_H, sizeof(uint8_t));
  return frame;
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

template void cpu_fill<uint8_t, 0>(uint8_t* dst, int width, int height, int pitch);
template void cpu_fill<uint16_t, 0>(uint16_t* dst, int width, int height, int pitch);

template <typename pixel_t, int fill_v>
__global__ void kl_fill(pixel_t* dst, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * pitch] = VHelper<pixel_t>::make(fill_v);
  }
}

template __global__ void kl_fill<uint8_t, 0>(uint8_t* dst, int width, int height, int pitch);
template __global__ void kl_fill<uint16_t, 0>(uint16_t* dst, int width, int height, int pitch);

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
void cpu_max(pixel_t* dst, const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto tmp = max(to_int(src0[x + y * pitch]), to_int(src1[x + y * pitch]));
      //dst[x + y * pitch] = VHelper<pixel_t>::cast_to(tmp);
      dst[x + y * pitch] = tmp;
    }
  }
}

template void cpu_max(uint8_t* dst, const uint8_t* __restrict__ src0, const uint8_t* __restrict__ src1, int width, int height, int pitch);
//template void cpu_max(uchar4* dst, const uchar4* __restrict__ src0, const uchar4* __restrict__ src1, int width, int height, int pitch);

template <typename pixel_t>
__global__ void kl_max(pixel_t* dst, const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto tmp = max(to_int(src0[x + y * pitch]), to_int(src1[x + y * pitch]));
    //dst[x + y * pitch] = VHelper<pixel_t>::cast_to(tmp);
    dst[x + y * pitch] = tmp;
  }
}

template __global__ void kl_max(uint8_t* dst, const uint8_t* __restrict__ src0, const uint8_t* __restrict__ src1, int width, int height, int pitch);
//template __global__ void kl_max(uchar4* dst, const uchar4* __restrict__ src0, const uchar4* __restrict__ src1, int width, int height, int pitch);

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
template void cpu_padv(uint16_t* dst, int width, int height, int pitch, int vpad);

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
template __global__ void kl_padv(uint16_t* dst, int width, int height, int pitch, int vpad);

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
template void cpu_padh(uint16_t* dst, int width, int height, int pitch, int hpad);

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
template __global__ void kl_padh(uint16_t* dst, int width, int height, int pitch, int hpad);

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
void KFMFilterBase::CopyFrame(Frame& src, Frame& dst, PNeoEnv env)
{
  const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
  const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
  const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
  pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
  pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
  pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

  int srcPitchY = src.GetPitch<pixel_t>(PLANAR_Y);
  int srcPitchUV = src.GetPitch<pixel_t>(PLANAR_U);
  int dstPitchY = dst.GetPitch<pixel_t>(PLANAR_Y);
  int dstPitchUV = dst.GetPitch<pixel_t>(PLANAR_U);

  int widthUV = srcvi.width >> logUVx;
  int heightUV = srcvi.height >> logUVy;

  Copy(dstY, dstPitchY, srcY, srcPitchY, srcvi.width, srcvi.height, env);
  Copy(dstU, dstPitchUV, srcU, srcPitchUV, widthUV, heightUV, env);
  Copy(dstV, dstPitchUV, srcV, srcPitchUV, widthUV, heightUV, env);
}

template void KFMFilterBase::CopyFrame<uint8_t>(Frame& src, Frame& dst, PNeoEnv env);
template void KFMFilterBase::CopyFrame<uint16_t>(Frame& src, Frame& dst, PNeoEnv env);


template <typename pixel_t>
void KFMFilterBase::PadFrame(Frame& dst, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

  vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
  vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
  vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

  int pitchY = dst.GetPitch<vpixel_t>(PLANAR_Y);
  int pitchUV = dst.GetPitch<vpixel_t>(PLANAR_U);
  int width4 = srcvi.width >> 2;
  int width4UV = width4 >> logUVx;
  int heightUV = srcvi.height >> logUVy;
  int vpadUV = VPAD >> logUVy;

  if (IS_CUDA) {
    dim3 threads(32, VPAD);
    dim3 blocks(nblocks(width4, threads.x));
    dim3 threadsUV(32, vpadUV);
    dim3 blocksUV(nblocks(width4UV, threads.x));
    kl_padv << <blocks, threads, 0, stream >> > (dstY, width4, srcvi.height, pitchY, VPAD);
    DEBUG_SYNC;
    kl_padv << <blocksUV, threadsUV, 0, stream >> > (dstU, width4UV, heightUV, pitchUV, vpadUV);
    DEBUG_SYNC;
    kl_padv << <blocksUV, threadsUV, 0, stream >> > (dstV, width4UV, heightUV, pitchUV, vpadUV);
    DEBUG_SYNC;
  }
  else {
    cpu_padv<vpixel_t>(dstY, width4, srcvi.height, pitchY, VPAD);
    cpu_padv<vpixel_t>(dstU, width4UV, heightUV, pitchUV, vpadUV);
    cpu_padv<vpixel_t>(dstV, width4UV, heightUV, pitchUV, vpadUV);
  }
}

template void KFMFilterBase::PadFrame<uint8_t>(Frame& dst, PNeoEnv env);
template void KFMFilterBase::PadFrame<uint16_t>(Frame& dst, PNeoEnv env);

template <typename vpixel_t>
void KFMFilterBase::LaunchAnalyzeFrame(uchar4* dst, int dstPitch,
  const vpixel_t* base, const vpixel_t* sref, const vpixel_t* mref,
  int width, int height, int pitch, int threshM, int threshS, int threshLS,
  PNeoEnv env)
{
  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_analyze_frame << <blocks, threads, 0, stream >> > (
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
void KFMFilterBase::AnalyzeFrame(Frame& f0, Frame& f1, Frame& flag,
  const FrameOldAnalyzeParam* prmY, const FrameOldAnalyzeParam* prmC, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;

  int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

  // 各プレーンを判定
  for (int pi = 0; pi < 3; ++pi) {
    int p = planes[pi];

    const vpixel_t* f0p = f0.GetReadPtr<vpixel_t>(p);
    const vpixel_t* f1p = f1.GetReadPtr<vpixel_t>(p);
    uchar4* flagp = flag.GetWritePtr<uchar4>(p);
    int pitch = f0.GetPitch<vpixel_t>(p);
    int dstPitch = flag.GetPitch<uchar4>(p);

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

template void KFMFilterBase::AnalyzeFrame<uint8_t>(Frame& f0, Frame& f1, Frame& flag,
  const FrameOldAnalyzeParam* prmY, const FrameOldAnalyzeParam* prmC, PNeoEnv env);
template void KFMFilterBase::AnalyzeFrame<uint16_t>(Frame& f0, Frame& f1, Frame& flag,
  const FrameOldAnalyzeParam* prmY, const FrameOldAnalyzeParam* prmC, PNeoEnv env);

void KFMFilterBase::MergeUVFlags(Frame& flag, PNeoEnv env)
{
  uint8_t* fY = flag.GetWritePtr<uint8_t>(PLANAR_Y);
  uint8_t* fU = flag.GetWritePtr<uint8_t>(PLANAR_U);
  uint8_t* fV = flag.GetWritePtr<uint8_t>(PLANAR_V);
  int pitchY = flag.GetPitch<uint8_t>(PLANAR_Y);
  int pitchUV = flag.GetPitch<uint8_t>(PLANAR_U);

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(srcvi.width, threads.x), nblocks(srcvi.height, threads.y));
    kl_merge_uvflags << <blocks, threads, 0, stream >> > (fY,
      fU, fV, srcvi.width, srcvi.height, pitchY, pitchUV, logUVx, logUVy);
    DEBUG_SYNC;
  }
  else {
    cpu_merge_uvflags(fY,
      fU, fV, srcvi.width, srcvi.height, pitchY, pitchUV, logUVx, logUVy);
  }
}

template <typename pixel_t>
void KFMFilterBase::MergeUVCoefs(Frame& flag, PNeoEnv env)
{
  pixel_t* fY = flag.GetWritePtr<pixel_t>(PLANAR_Y);
  pixel_t* fU = flag.GetWritePtr<pixel_t>(PLANAR_U);
  pixel_t* fV = flag.GetWritePtr<pixel_t>(PLANAR_V);
  int pitchY = flag.GetPitch<pixel_t>(PLANAR_Y);
  int pitchUV = flag.GetPitch<pixel_t>(PLANAR_U);

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
    kl_merge_uvcoefs << <blocks, threads, 0, stream >> > (fY,
      fU, fV, vi.width, vi.height, pitchY, pitchUV, logUVx, logUVy);
    DEBUG_SYNC;
  }
  else {
    cpu_merge_uvcoefs(fY,
      fU, fV, vi.width, vi.height, pitchY, pitchUV, logUVx, logUVy);
  }
}

template void KFMFilterBase::MergeUVCoefs<uint8_t>(Frame& flag, PNeoEnv env);
template void KFMFilterBase::MergeUVCoefs<uint16_t>(Frame& flag, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::ApplyUVCoefs(Frame& flag, PNeoEnv env)
{
  pixel_t* fY = flag.GetWritePtr<pixel_t>(PLANAR_Y);
  pixel_t* fU = flag.GetWritePtr<pixel_t>(PLANAR_U);
  pixel_t* fV = flag.GetWritePtr<pixel_t>(PLANAR_V);
  int pitchY = flag.GetPitch<pixel_t>(PLANAR_Y);
  int pitchUV = flag.GetPitch<pixel_t>(PLANAR_U);
  int widthUV = vi.width >> logUVx;
  int heightUV = vi.height >> logUVy;

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
    kl_apply_uvcoefs_420 << <blocks, threads, 0, stream >> > (fY,
      fU, fV, widthUV, heightUV, pitchY, pitchUV);
    DEBUG_SYNC;
  }
  else {
    cpu_apply_uvcoefs_420(fY, fU, fV, widthUV, heightUV, pitchY, pitchUV);
  }
}

template void KFMFilterBase::ApplyUVCoefs<uint8_t>(Frame& flag, PNeoEnv env);
template void KFMFilterBase::ApplyUVCoefs<uint16_t>(Frame& flag, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::ExtendCoefs(Frame& src, Frame& dst, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
  const vpixel_t* srcY = src.GetReadPtr<vpixel_t>(PLANAR_Y);
  vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);

  int pitchY = src.GetPitch<vpixel_t>(PLANAR_Y);
  int width4 = vi.width >> 2;

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
    kl_extend_coef << <blocks, threads, 0, stream >> > (
      dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
    DEBUG_SYNC;
    dim3 threadsB(32, 1);
    dim3 blocksB(nblocks(width4, threads.x));
    kl_copy_border << <blocksB, threadsB, 0, stream >> > (
      dstY, srcY, width4, vi.height, pitchY, 1);
    DEBUG_SYNC;
  }
  else {
    cpu_extend_coef(dstY + pitchY, srcY + pitchY, width4, vi.height - 2, pitchY);
    cpu_copy_border(dstY, srcY, width4, vi.height, pitchY, 1);
  }
}

template void KFMFilterBase::ExtendCoefs<uint8_t>(Frame& src, Frame& dst, PNeoEnv env);
template void KFMFilterBase::ExtendCoefs<uint16_t>(Frame& src, Frame& dst, PNeoEnv env);

template <typename pixel_t>
void KFMFilterBase::CompareFields(Frame& src, Frame& flag, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

  const vpixel_t* srcY = src.GetReadPtr<vpixel_t>(PLANAR_Y);
  const vpixel_t* srcU = src.GetReadPtr<vpixel_t>(PLANAR_U);
  const vpixel_t* srcV = src.GetReadPtr<vpixel_t>(PLANAR_V);
  vpixel_t* dstY = flag.GetWritePtr<vpixel_t>(PLANAR_Y);
  vpixel_t* dstU = flag.GetWritePtr<vpixel_t>(PLANAR_U);
  vpixel_t* dstV = flag.GetWritePtr<vpixel_t>(PLANAR_V);

  int pitchY = src.GetPitch<vpixel_t>(PLANAR_Y);
  int pitchUV = src.GetPitch<vpixel_t>(PLANAR_U);
  int width4 = vi.width >> 2;
  int width4UV = width4 >> logUVx;
  int heightUV = vi.height >> logUVy;

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
    dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
    kl_calc_combe << <blocks, threads, 0, stream >> > (dstY, srcY, width4, vi.height, pitchY);
    DEBUG_SYNC;
    kl_calc_combe << <blocksUV, threads, 0, stream >> > (dstU, srcU, width4UV, heightUV, pitchUV);
    DEBUG_SYNC;
    kl_calc_combe << <blocksUV, threads, 0, stream >> > (dstV, srcV, width4UV, heightUV, pitchUV);
    DEBUG_SYNC;
  }
  else {
    cpu_calc_combe(dstY, srcY, width4, vi.height, pitchY);
    cpu_calc_combe(dstU, srcU, width4UV, heightUV, pitchUV);
    cpu_calc_combe(dstV, srcV, width4UV, heightUV, pitchUV);
  }
}

template void KFMFilterBase::CompareFields<uint8_t>(Frame& src, Frame& flag, PNeoEnv env);
template void KFMFilterBase::CompareFields<uint16_t>(Frame& src, Frame& flag, PNeoEnv env);

template <typename pixel_t>
void cpu_max_extend_blocks(pixel_t* dstp, int pitch, int nBlkX, int nBlkY)
{
  for (int by = 1; by < nBlkY; ++by) {
    dstp[0 + by * pitch] = dstp[0 + 1 + (by + 0) * pitch];
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

template <typename pixel_t>
__global__ void kl_max_extend_blocks_h(pixel_t* dstp, const pixel_t* srcp, int pitch, int nBlkX, int nBlkY)
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

template <typename pixel_t>
__global__ void kl_max_extend_blocks_v(pixel_t* dstp, const pixel_t* srcp, int pitch, int nBlkX, int nBlkY)
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
void KFMFilterBase::ExtendBlocks(Frame& dst, Frame& tmp, bool uv, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

  pixel_t* tmpY = tmp.GetWritePtr<pixel_t>(PLANAR_Y);
  pixel_t* tmpU = tmp.GetWritePtr<pixel_t>(PLANAR_U);
  pixel_t* tmpV = tmp.GetWritePtr<pixel_t>(PLANAR_V);
  pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
  pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
  pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

  int pitchY = tmp.GetPitch<pixel_t>(PLANAR_Y);
  int pitchUV = tmp.GetPitch<pixel_t>(PLANAR_U);
  int width = tmp.GetWidth<pixel_t>(PLANAR_Y);
  int widthUV = tmp.GetWidth<pixel_t>(PLANAR_U);
  int height = tmp.GetHeight(PLANAR_Y);
  int heightUV = tmp.GetHeight(PLANAR_U);

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
    kl_max_extend_blocks_h << <blocks, threads, 0, stream >> > (tmpY, dstY, pitchY, width, height);
    kl_max_extend_blocks_v << <blocks, threads, 0, stream >> > (dstY, tmpY, pitchY, width, height);
    DEBUG_SYNC;
    if (uv) {
      kl_max_extend_blocks_h << <blocksUV, threads, 0, stream >> > (tmpU, dstU, pitchUV, widthUV, heightUV);
      kl_max_extend_blocks_v << <blocksUV, threads, 0, stream >> > (dstU, tmpU, pitchUV, widthUV, heightUV);
      DEBUG_SYNC;
      kl_max_extend_blocks_h << <blocksUV, threads, 0, stream >> > (tmpV, dstV, pitchUV, widthUV, heightUV);
      kl_max_extend_blocks_v << <blocksUV, threads, 0, stream >> > (dstV, tmpV, pitchUV, widthUV, heightUV);
      DEBUG_SYNC;
    }
  }
  else {
    cpu_max_extend_blocks(dstY, pitchY, width, height);
    if (uv) {
      cpu_max_extend_blocks(dstU, pitchUV, widthUV, heightUV);
      cpu_max_extend_blocks(dstV, pitchUV, widthUV, heightUV);
    }
  }
}

template void KFMFilterBase::ExtendBlocks<uint8_t>(Frame& dst, Frame& tmp, bool uv, PNeoEnv env);
template void KFMFilterBase::ExtendBlocks<uchar4>(Frame& dst, Frame& tmp, bool uv, PNeoEnv env);

template <typename vpixel_t, typename fpixel_t>
void cpu_merge(vpixel_t* dst,
  const vpixel_t* src24, const vpixel_t* src60,
  int width, int height, int pitch,
  const fpixel_t* flagp, int fpitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 combe = to_int(flagp[x + y * fpitch]);
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
  const fpixel_t* flagp, int fpitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 combe = to_int(flagp[x + y * fpitch]);
    int4 invcombe = VHelper<int4>::make(128) - combe;
    int4 tmp = (combe * to_int(src60[x + y * pitch]) + invcombe * to_int(src24[x + y * pitch]) + 64) >> 7;
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

template <typename pixel_t>
void KFMFilterBase::MergeBlock(Frame& src24, Frame& src60, Frame& flag, Frame& dst, PNeoEnv env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

  const vpixel_t* src24Y = src24.GetReadPtr<vpixel_t>(PLANAR_Y);
  const vpixel_t* src24U = src24.GetReadPtr<vpixel_t>(PLANAR_U);
  const vpixel_t* src24V = src24.GetReadPtr<vpixel_t>(PLANAR_V);
  const vpixel_t* src60Y = src60.GetReadPtr<vpixel_t>(PLANAR_Y);
  const vpixel_t* src60U = src60.GetReadPtr<vpixel_t>(PLANAR_U);
  const vpixel_t* src60V = src60.GetReadPtr<vpixel_t>(PLANAR_V);
  vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
  vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
  vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);
  const uchar4* flagY = flag.GetReadPtr<uchar4>(PLANAR_Y);
  const uchar4* flagC = flag.GetReadPtr<uchar4>(PLANAR_U);

  int pitchY = src24.GetPitch<vpixel_t>(PLANAR_Y);
  int pitchUV = src24.GetPitch<vpixel_t>(PLANAR_U);
  int width4 = vi.width >> 2;
  int width4UV = width4 >> logUVx;
  int heightUV = vi.height >> logUVy;
  int fpitchY = flag.GetPitch<uchar4>(PLANAR_Y);
  int fpitchUV = flag.GetPitch<uchar4>(PLANAR_U);

  if (IS_CUDA) {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
    dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
    kl_merge << <blocks, threads, 0, stream >> > (
      dstY, src24Y, src60Y, width4, vi.height, pitchY, flagY, fpitchY);
    DEBUG_SYNC;
    kl_merge << <blocksUV, threads, 0, stream >> > (
      dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagC, fpitchUV);
    DEBUG_SYNC;
    kl_merge << <blocksUV, threads, 0, stream >> > (
      dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagC, fpitchUV);
    DEBUG_SYNC;
  }
  else {
    cpu_merge(dstY, src24Y, src60Y, width4, vi.height, pitchY, flagY, fpitchY);
    cpu_merge(dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagC, fpitchUV);
    cpu_merge(dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagC, fpitchUV);
  }
}

template void KFMFilterBase::MergeBlock<uint8_t>(Frame& src24, Frame& src60, Frame& flag, Frame& dst, PNeoEnv env);
template void KFMFilterBase::MergeBlock<uint16_t>(Frame& src24, Frame& src60, Frame& flag, Frame& dst, PNeoEnv env);

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
