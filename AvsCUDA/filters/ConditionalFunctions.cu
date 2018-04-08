#include "avisynth.h"

#include <limits>
#include <memory>

#include "../AvsCUDA.h"
#include "avs/alignment.h"
#include "focus.h"

#include "CommonFunctions.h"
#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"

#pragma region AveragePlane CUDA
enum {
  SUM_TH_W = 16,
  SUM_TH_H = 16,
  SUM_THREADS = SUM_TH_W * SUM_TH_H
};

template <typename T>
__global__ void kl_init_sum(T* sum)
{
  sum[threadIdx.x] = 0;
}

__device__ uchar4 clamp_to_range(uchar4 src, int maxv) { return src; }
__device__ int4 clamp_to_range(ushort4 src, int maxv) { return min(to_int(src), maxv); }
__device__ float4 clamp_to_range(float4 src, int maxv) { return src; }

template <typename vpixel_t, typename gsum_t, typename lsum_t>
__global__ void kl_sum_of_pixels(const vpixel_t* __restrict__ src, int width, int height, int pitch, int maxv, gsum_t* sum)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  lsum_t tmpsum = lsum_t();
  if (x < width && y < height) {
    auto s = clamp_to_range(src[x + y * pitch], maxv);
    tmpsum = s.x + s.y + s.z + s.w;
  }

  __shared__ lsum_t sbuf[SUM_THREADS];
  dev_reduce<lsum_t, SUM_THREADS, AddReducer<lsum_t>>(tid, tmpsum, sbuf);

  if (tid == 0) {
    atomicAdd(sum, (gsum_t)tmpsum);
  }
}

template <typename vpixel_t, typename gsum_t, typename lsum_t>
double calc_sum_of_pixels(const void* src, int width, int height, int pitch, int maxv, void* sum, PNeoEnv env)
{
  int width4 = width >> 2;
  int pitch4 = pitch >> 2;

  kl_init_sum << <1, 1 >> > ((gsum_t*)sum);
  DEBUG_SYNC;

  dim3 threads(SUM_TH_W, SUM_TH_H);
  dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
  kl_sum_of_pixels <vpixel_t, gsum_t, lsum_t><< <blocks, threads >> > (
    (const vpixel_t*)src, width4, height, pitch4, maxv, (gsum_t*)sum);
  DEBUG_SYNC;

  gsum_t ret;
  CUDA_CHECK(cudaMemcpy(&ret, sum, sizeof(gsum_t), cudaMemcpyDeviceToHost));
  return ((double)ret / (height * width));;
}
#pragma endregion

#pragma region AveragePlane CPU

// Average plane
template<typename pixel_t>
static double get_sum_of_pixels_c(const BYTE* srcp8, size_t height, size_t width, size_t pitch) {
  typedef typename std::conditional < sizeof(pixel_t) == 4, double, __int64>::type sum_t;
  sum_t accum = 0; // int32 holds sum of maximum 16 Mpixels for 8 bit, and 65536 pixels for uint16_t pixels
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  pitch /= sizeof(pixel_t);
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      accum += srcp[x];
    }
    srcp += pitch;
  }
  return (double)accum;
}

// sum: sad with zero
static double get_sum_of_pixels_sse2(const BYTE* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod16_width = width / 16 * 16;
  __int64 result = 0;
  __m128i sum = _mm_setzero_si128();
  __m128i zero = _mm_setzero_si128();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i sad = _mm_sad_epu8(src, zero);
      sum = _mm_add_epi32(sum, sad);
    }

    for (size_t x = mod16_width; x < width; ++x) {
      result += srcp[x];
    }

    srcp += pitch;
  }
  __m128i upper = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
  sum = _mm_add_epi32(sum, upper);
  result += _mm_cvtsi128_si32(sum);
  return (double)result;
}

#ifdef X86_32
static double get_sum_of_pixels_isse(const BYTE* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod8_width = width / 8 * 8;
  __int64 result = 0;
  __m64 sum = _mm_setzero_si64();
  __m64 zero = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 sad = _mm_sad_pu8(src, zero);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += srcp[x];
    }

    srcp += pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return (double)result;
}
#endif

#pragma endregion

#pragma region AveragePlane
class AveragePlane {

public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return AvgPlane(args[0], user_data, plane, args[1].AsInt(0), env);
  }
  static AVSValue AvgPlane(AVSValue clip, void* user_data, int plane, int offset, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!clip.IsClip())
      env->ThrowError("Average Plane: No clip supplied!");

    PClip child = clip.AsClip();
    VideoInfo vi = child->GetVideoInfo();

    if (!vi.IsPlanar())
      env->ThrowError("Average Plane: Only planar YUV or planar RGB images supported!");

    AVSValue cn = env->GetVarDef("current_frame");
    if (!cn.IsInt())
      env->ThrowError("Average Plane: This filter can only be used within run-time filters");

    int n = cn.AsInt();
    n = min(max(n + offset, 0), vi.num_frames - 1);

    PVideoFrame src = child->GetFrame(n, env);

    int pixelsize = vi.ComponentSize();

    const BYTE* srcp = src->GetReadPtr(plane);
    int height = src->GetHeight(plane);
    int width = src->GetRowSize(plane) / pixelsize;

    if (width == 0 || height == 0)
      env->ThrowError("Average Plane: plane does not exist!");

    if (IS_CUDA) {
      // CUDA
      int pitch = src->GetPitch(plane) / pixelsize;

      if (width % 4)
        env->ThrowError("Average Plane: width must be multiple of 4 on CUDA");

      int bits_per_pixel = vi.BitsPerComponent();
      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case
        sum_in_32bits = ((__int64)total_pixels * (__int64(1) << bits_per_pixel)) <= std::numeric_limits<int>::max();

      VideoInfo workvi = VideoInfo();
      workvi.pixel_type = VideoInfo::CS_BGR32;
      workvi.width = 4;
      workvi.height = 1; // 16bytes
      PVideoFrame work = env->NewVideoFrame(workvi);
      void* workbuf = work->GetWritePtr();

      int maxv = ((1 << bits_per_pixel) - 1);

      switch (pixelsize) {
      case 1:
        if (sum_in_32bits)
          return calc_sum_of_pixels<uchar4, uint32_t, uint32_t>(srcp, width, height, pitch, maxv, workbuf, env);
        else
          return calc_sum_of_pixels<uchar4, uint64_t, int>(srcp, width, height, pitch, maxv, workbuf, env);
      case 2:
        if (sum_in_32bits)
          return calc_sum_of_pixels<ushort4, uint32_t, uint32_t>(srcp, width, height, pitch, maxv, workbuf, env);
        else
          return calc_sum_of_pixels<ushort4, uint64_t, uint32_t>(srcp, width, height, pitch, maxv, workbuf, env);
      case 4:
        return calc_sum_of_pixels<float4, float, float>(srcp, width, height, pitch, maxv, workbuf, env);
      }
    }
    else {
      // CPU
      int pitch = src->GetPitch(plane);

      double sum = 0.0;

      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case
        sum_in_32bits = ((__int64)total_pixels * (pixelsize == 1 ? 255 : 65535)) <= std::numeric_limits<int>::max();

      if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && width >= 16) {
        sum = get_sum_of_pixels_sse2(srcp, height, width, pitch);
      }
      else
#ifdef X86_32
        if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
          sum = get_sum_of_pixels_isse(srcp, height, width, pitch);
        }
        else
#endif
        {
          if (pixelsize == 1)
            sum = get_sum_of_pixels_c<uint8_t>(srcp, height, width, pitch);
          else if (pixelsize == 2)
            sum = get_sum_of_pixels_c<uint16_t>(srcp, height, width, pitch);
          else // pixelsize==4
            sum = get_sum_of_pixels_c<float>(srcp, height, width, pitch);
        }

      float f = (float)(sum / (height * width));

      return (AVSValue)f;
    }

    return (AVSValue)0;
  }
};
#pragma endregion

#pragma region ComparePlane CUDA

__device__ int4 diff_pixel(uchar4 src0, uchar4 src1, int maxv) {
  return absdiff(src0, src1);
}
__device__ int4 diff_pixel(ushort4 src0, ushort4 src1, int maxv) {
  return abs(min(to_int(src0), maxv) - min(to_int(src1), maxv));
}
__device__ float4 diff_pixel(float4 src0, float4 src1, int maxv) {
  return abs(src0 - src1);
}

template <typename vpixel_t, typename gsum_t, typename lsum_t>
__global__ void kl_sad(
  const vpixel_t* __restrict__ src0,
  const vpixel_t* __restrict__ src1, 
  int width, int height, int pitch, int maxv, gsum_t* sum)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  lsum_t tmpsum = lsum_t();
  if (x < width && y < height) {
    auto s = diff_pixel(src0[x + y * pitch], src1[x + y * pitch], maxv);
    tmpsum = s.x + s.y + s.z + s.w;
  }

  __shared__ lsum_t sbuf[SUM_THREADS];
  dev_reduce<lsum_t, SUM_THREADS, AddReducer<lsum_t>>(tid, tmpsum, sbuf);

  if (tid == 0) {
    atomicAdd(sum, (gsum_t)tmpsum);
  }
}

template <typename vpixel_t, typename gsum_t, typename lsum_t>
double calc_sad(const void* src0, const void* src1, int width, int height, int pitch, int maxv, void* sum, bool is_rgb, PNeoEnv env)
{
  int width4 = width >> 2;
  int pitch4 = pitch >> 2;

  kl_init_sum << <1, 1 >> > ((gsum_t*)sum);
  DEBUG_SYNC;

  dim3 threads(SUM_TH_W, SUM_TH_H);
  dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
  kl_sad <vpixel_t, gsum_t, lsum_t> << <blocks, threads >> > (
    (const vpixel_t*)src0, (const vpixel_t*)src1, width4, height, pitch4, maxv, (gsum_t*)sum);
  DEBUG_SYNC;

  gsum_t sad;
  CUDA_CHECK(cudaMemcpy(&sad, sum, sizeof(gsum_t), cudaMemcpyDeviceToHost));

  if (is_rgb)
    return (((double)sad * 4) / (height * width * 3)); // why * 4/3? alpha plane was masked out, anyway
  else
    return ((double)sad / (height * width));
}

#pragma endregion

#pragma region ComparePlane CPU

template<typename pixel_t>
static double get_sad_c(const BYTE* c_plane8, const BYTE* t_plane8, size_t height, size_t width, size_t c_pitch, size_t t_pitch) {
  const pixel_t *c_plane = reinterpret_cast<const pixel_t *>(c_plane8);
  const pixel_t *t_plane = reinterpret_cast<const pixel_t *>(t_plane8);
  c_pitch /= sizeof(pixel_t);
  t_pitch /= sizeof(pixel_t);
  typedef typename std::conditional < sizeof(pixel_t) == 4, double, __int64>::type sum_t;
  sum_t accum = 0; // int32 holds sum of maximum 16 Mpixels for 8 bit, and 65536 pixels for uint16_t pixels

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      accum += std::abs(t_plane[x] - c_plane[x]);
    }
    c_plane += c_pitch;
    t_plane += t_pitch;
  }
  return (double)accum;

}

template<typename pixel_t>
static double get_sad_rgb_c(const BYTE* c_plane8, const BYTE* t_plane8, size_t height, size_t width, size_t c_pitch, size_t t_pitch) {
  const pixel_t *c_plane = reinterpret_cast<const pixel_t *>(c_plane8);
  const pixel_t *t_plane = reinterpret_cast<const pixel_t *>(t_plane8);
  c_pitch /= sizeof(pixel_t);
  t_pitch /= sizeof(pixel_t);
  __int64 accum = 0; // packed rgb: integer type only
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x += 4) {
      accum += std::abs(t_plane[x] - c_plane[x]);
      accum += std::abs(t_plane[x + 1] - c_plane[x + 1]);
      accum += std::abs(t_plane[x + 2] - c_plane[x + 2]);
    }
    c_plane += c_pitch;
    t_plane += t_pitch;
  }
  return (double)accum;

}

#ifdef X86_32

static size_t get_sad_isse(const BYTE* src_ptr, const BYTE* other_ptr, size_t height, size_t width, size_t src_pitch, size_t other_pitch) {
  size_t mod8_width = width / 8 * 8;
  size_t result = 0;
  __m64 sum = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(src_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      __m64 sad = _mm_sad_pu8(src, other);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += std::abs(src_ptr[x] - other_ptr[x]);
    }

    src_ptr += src_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}

static size_t get_sad_rgb_isse(const BYTE* src_ptr, const BYTE* other_ptr, size_t height, size_t width, size_t src_pitch, size_t other_pitch) {
  size_t mod8_width = width / 8 * 8;
  size_t result = 0;
  __m64 rgb_mask = _mm_set1_pi32(0x00FFFFFF);
  __m64 sum = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(src_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      src = _mm_and_si64(src, rgb_mask);
      other = _mm_and_si64(other, rgb_mask);
      __m64 sad = _mm_sad_pu8(src, other);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += std::abs(src_ptr[x] - other_ptr[x]);
    }

    src_ptr += src_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}

#endif

#pragma endregion

#pragma region ComparePlane
class ComparePlane {

public:
  static AVSValue CmpPlane(AVSValue clip, AVSValue clip2, void* user_data, int plane, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!clip.IsClip())
      env->ThrowError("Plane Difference: No clip supplied!");
    if (!clip2.IsClip())
      env->ThrowError("Plane Difference: Second parameter is not a clip!");

    PClip child = clip.AsClip();
    VideoInfo vi = child->GetVideoInfo();
    PClip child2 = clip2.AsClip();
    VideoInfo vi2 = child2->GetVideoInfo();
    if (plane != -1) {
      if (!vi.IsPlanar() || !vi2.IsPlanar())
        env->ThrowError("Plane Difference: Only planar YUV or planar RGB images supported!");
    }
    else {
      if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
        env->ThrowError("RGB Difference: Planar RGB is not supported here (clip 1)");
      if (vi2.IsPlanarRGB() || vi2.IsPlanarRGBA())
        env->ThrowError("RGB Difference: Planar RGB is not supported here (clip 2)");
      if (!vi.IsRGB())
        env->ThrowError("RGB Difference: RGB difference can only be tested on RGB images! (clip 1)");
      if (!vi2.IsRGB())
        env->ThrowError("RGB Difference: RGB difference can only be tested on RGB images! (clip 2)");
      plane = 0;
    }

    AVSValue cn = env->GetVarDef("current_frame");
    if (!cn.IsInt())
      env->ThrowError("Plane Difference: This filter can only be used within run-time filters");

    int n = cn.AsInt();
    n = clamp(n, 0, vi.num_frames - 1);

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame src2 = child2->GetFrame(n, env);

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    const BYTE* srcp = src->GetReadPtr(plane);
    const BYTE* srcp2 = src2->GetReadPtr(plane);
    const int height = src->GetHeight(plane);
    const int rowsize = src->GetRowSize(plane);
    const int width = rowsize / pixelsize;
    const int height2 = src2->GetHeight(plane);
    const int rowsize2 = src2->GetRowSize(plane);
    const int width2 = rowsize2 / pixelsize;

    if (vi.ComponentSize() != vi2.ComponentSize())
      env->ThrowError("Plane Difference: Bit-depth are not the same!");

    if (width == 0 || height == 0)
      env->ThrowError("Plane Difference: plane does not exist!");

    if (height != height2 || width != width2)
      env->ThrowError("Plane Difference: Images are not the same size!");

    if (IS_CUDA) {
      // CUDA
      const int pitch = src->GetPitch(plane) / pixelsize;
      const int pitch2 = src2->GetPitch(plane) / pixelsize;

      if (width % 4)
        env->ThrowError("Plane Difference: width must be multiple of 4 on CUDA");

      if (pitch != pitch2)
        env->ThrowError("Plane Difference: pitch must be same on CUDA!");

      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case check
        sum_in_32bits = ((__int64)total_pixels * ((__int64(1) << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();

      VideoInfo workvi = VideoInfo();
      workvi.pixel_type = VideoInfo::CS_BGR32;
      workvi.width = 4;
      workvi.height = 1; // 16bytes
      PVideoFrame work = env->NewVideoFrame(workvi);
      void* workbuf = work->GetWritePtr();

      int maxv = ((1 << bits_per_pixel) - 1);
      bool is_rgb = (vi.IsRGB32() || vi.IsRGB64());

      switch (pixelsize) {
      case 1:
        if (sum_in_32bits)
          return calc_sad<uchar4, uint32_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
        else
          return calc_sad<uchar4, uint64_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      case 2:
        if (sum_in_32bits)
          return calc_sad<ushort4, uint32_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
        else
          return calc_sad<ushort4, uint64_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      case 4:
        return calc_sad<float4, float, float>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      }

      return (AVSValue)0;
    }
    else {
      // CPU
      const int pitch = src->GetPitch(plane);
      const int pitch2 = src2->GetPitch(plane);
      
      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case check
        sum_in_32bits = ((__int64)total_pixels * ((1 << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();

      double sad = 0.0;

      // for c: width, for sse: rowsize
      if (vi.IsRGB32() || vi.IsRGB64()) {
        if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          // int64 internally, no sum_in_32bits
          sad = (double)calculate_sad_8_or_16_sse2<uint16_t, true>(srcp, srcp2, pitch, pitch2, width*pixelsize, height); // in focus. 21.68/21.39
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          sad = (double)calculate_sad_8_or_16_sse2<uint8_t, true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
        }
        else
#ifdef X86_32
          if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
            sad = get_sad_rgb_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
          }
          else
#endif
          {
            if (pixelsize == 1)
              sad = (double)get_sad_rgb_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
            else // pixelsize==2
              sad = (double)get_sad_rgb_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
          }
      }
      else {
        if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          sad = (double)calculate_sad_8_or_16_sse2<uint16_t, false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
        }
        else
          if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
            sad = (double)calculate_sad_8_or_16_sse2<uint8_t, false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
          }
          else
#ifdef X86_32
            if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
              sad = get_sad_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
            }
            else
#endif
            {
              if (pixelsize == 1)
                sad = get_sad_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
              else if (pixelsize == 2)
                sad = get_sad_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
              else // pixelsize==4
                sad = get_sad_c<float>(srcp, srcp2, height, width, pitch, pitch2);
            }
      }

      float f;

      if (vi.IsRGB32() || vi.IsRGB64())
        f = (float)((sad * 4) / (height * width * 3)); // why * 4/3? alpha plane was masked out, anyway
      else
        f = (float)(sad / (height * width));

      return (AVSValue)f;
    }
  }
  static AVSValue CmpPlaneSame(AVSValue clip, void* user_data, int offset, int plane, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!clip.IsClip())
      env->ThrowError("Plane Difference: No clip supplied!");

    PClip child = clip.AsClip();
    VideoInfo vi = child->GetVideoInfo();
    if (plane == -1) {
      if (!vi.IsRGB() || vi.IsPlanarRGB() || vi.IsPlanarRGBA())
        env->ThrowError("RGB Difference: RGB difference can only be calculated on packed RGB images");
      plane = 0;
    }
    else {
      if (!vi.IsPlanar())
        env->ThrowError("Plane Difference: Only planar YUV or planar RGB images images supported!");
    }

    AVSValue cn = env->GetVarDef("current_frame");
    if (!cn.IsInt())
      env->ThrowError("Plane Difference: This filter can only be used within run-time filters");

    int n = cn.AsInt();
    n = clamp(n, 0, vi.num_frames - 1);
    int n2 = clamp(n + offset, 0, vi.num_frames - 1);

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame src2 = child->GetFrame(n2, env);

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    const BYTE* srcp = src->GetReadPtr(plane);
    const BYTE* srcp2 = src2->GetReadPtr(plane);
    int height = src->GetHeight(plane);
    int rowsize = src->GetRowSize(plane);
    int width = rowsize / pixelsize;

    if (width == 0 || height == 0)
      env->ThrowError("Plane Difference: No chroma planes in greyscale clip!");

    if (IS_CUDA) {
      // CUDA
      int pitch = src->GetPitch(plane) / pixelsize;
      int pitch2 = src2->GetPitch(plane) / pixelsize;

      if (width % 4)
        env->ThrowError("Plane Difference: width must be multiple of 4 on CUDA");

      if (width % 4)
        env->ThrowError("Plane Difference: width must be multiple of 4 on CUDA");

      if (pitch != pitch2)
        env->ThrowError("Plane Difference: pitch must be same on CUDA!");

      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case check
        sum_in_32bits = ((__int64)total_pixels * ((__int64(1) << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();

      VideoInfo workvi = VideoInfo();
      workvi.pixel_type = VideoInfo::CS_BGR32;
      workvi.width = 4;
      workvi.height = 1; // 16bytes
      PVideoFrame work = env->NewVideoFrame(workvi);
      void* workbuf = work->GetWritePtr();

      int maxv = ((1 << bits_per_pixel) - 1);
      bool is_rgb = (vi.IsRGB32() || vi.IsRGB64());

      switch (pixelsize) {
      case 1:
        if (sum_in_32bits)
          return calc_sad<uchar4, uint32_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
        else
          return calc_sad<uchar4, uint64_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      case 2:
        if (sum_in_32bits)
          return calc_sad<ushort4, uint32_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
        else
          return calc_sad<ushort4, uint64_t, uint32_t>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      case 4:
        return calc_sad<float4, float, float>(srcp, srcp2, width, height, pitch, maxv, workbuf, is_rgb, env);
      }

      return (AVSValue)0;
    }
    else {
      // CPU
      int pitch = src->GetPitch(plane);
      int pitch2 = src2->GetPitch(plane);

      int total_pixels = width * height;
      bool sum_in_32bits;
      if (pixelsize == 4)
        sum_in_32bits = false;
      else // worst case check
        sum_in_32bits = ((__int64)total_pixels * ((1 << bits_per_pixel) - 1)) <= std::numeric_limits<int>::max();

      double sad = 0;
      // for c: width, for sse: rowsize
      if (vi.IsRGB32() || vi.IsRGB64()) {
        if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          // int64 internally, no sum_in_32bits
          sad = (double)calculate_sad_8_or_16_sse2<uint16_t, true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus. 21.68/21.39
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          sad = (double)calculate_sad_8_or_16_sse2<uint8_t, true>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
        }
        else
#ifdef X86_32
          if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
            sad = get_sad_rgb_isse(srcp, srcp2, height, rowsize, pitch, pitch2);
          }
          else
#endif
          {
            if (pixelsize == 1)
              sad = get_sad_rgb_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
            else
              sad = get_sad_rgb_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
          }
      }
      else {
        if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          sad = (double)calculate_sad_8_or_16_sse2<uint16_t, false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16) && IsPtrAligned(srcp2, 16) && rowsize >= 16) {
          sad = (double)calculate_sad_8_or_16_sse2<uint8_t, false>(srcp, srcp2, pitch, pitch2, rowsize, height); // in focus, no overflow
        }
        else
#ifdef X86_32
          if ((pixelsize == 1) && sum_in_32bits && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && width >= 8) {
            sad = get_sad_isse(srcp, srcp2, height, width, pitch, pitch2);
          }
          else
#endif
          {
            if (pixelsize == 1)
              sad = get_sad_c<uint8_t>(srcp, srcp2, height, width, pitch, pitch2);
            else if (pixelsize == 2)
              sad = get_sad_c<uint16_t>(srcp, srcp2, height, width, pitch, pitch2);
            else // pixelsize==4
              sad = get_sad_c<float>(srcp, srcp2, height, width, pitch, pitch2);
          }
      }

      float f;

      if (vi.IsRGB32() || vi.IsRGB64())
        f = (float)((sad * 4) / (height * width * 3));
      else
        f = (float)(sad / (height * width));

      return (AVSValue)f;
    }
  }

  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return CmpPlane(args[0], args[1], user_data, plane, env);
  }
  static AVSValue Create_prev(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return CmpPlaneSame(args[0], user_data, -1, plane, env);
  }
  static AVSValue Create_next(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return CmpPlaneSame(args[0], user_data, args[1].AsInt(1), plane, env);
  }
};
#pragma endregion

#pragma region MinMaxPlane CUDA
template <typename T>
__global__ void kl_init_hist(T* sum, int len)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if (x < len) {
    sum[x] = 0;
  }
}

__device__ uchar4 to_count_index(uchar4 src, int maxv) { return src; }
__device__ int4 to_count_index(ushort4 src, int maxv) { return min(to_int(src), maxv); }
__device__ int4 to_count_index(float4 src, int maxv) { return to_int(clamp(src * 65535.0f + 0.5f, 0.0f, 65535.0f)); }

template <typename vpixel_t, typename sum_t>
__global__ void kl_count_hist(
  const vpixel_t* __restrict__ src,
  int width, int height, int pitch, int maxv, sum_t* sum)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto s = to_count_index(src[x + y * pitch], maxv);
    atomicAdd(&sum[s.x], 1);
    atomicAdd(&sum[s.y], 1);
    atomicAdd(&sum[s.z], 1);
    atomicAdd(&sum[s.w], 1);
  }
}

template <typename vpixel_t>
void calc_count_hist(
  const void* src, int width, int height, int pitch, int maxv,
  int* dev_sum, int* host_sum, int length,
  PNeoEnv env)
{
  int width4 = width >> 2;
  int pitch4 = pitch >> 2;

  kl_init_hist << <16, nblocks(length, 16) >> > (dev_sum, length);
  DEBUG_SYNC;

  dim3 threads(SUM_TH_W, SUM_TH_H);
  dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
  kl_count_hist << <blocks, threads >> > (
    (const vpixel_t*)src, width4, height, pitch4, maxv, dev_sum);
  DEBUG_SYNC;

  CUDA_CHECK(cudaMemcpy(host_sum, dev_sum, sizeof(int) * length, cudaMemcpyDeviceToHost));
}
#pragma endregion

#pragma region MinMaxPlane
class MinMaxPlane {

public:
  static AVSValue MinMax(AVSValue clip, void* user_data, double threshold, int offset, int plane, int mode, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!clip.IsClip())
      env->ThrowError("MinMax: No clip supplied!");

    PClip child = clip.AsClip();
    VideoInfo vi = child->GetVideoInfo();

    if (!vi.IsPlanar())
      env->ThrowError("MinMax: Image must be planar");

    int pixelsize = vi.ComponentSize();

    // Get current frame number
    AVSValue cn = env->GetVarDef("current_frame");
    if (!cn.IsInt())
      env->ThrowError("MinMax: This filter can only be used within run-time filters");

    int n = cn.AsInt();
    n = min(max(n + offset, 0), vi.num_frames - 1);

    // Prepare the source
    PVideoFrame src = child->GetFrame(n, env);

    const BYTE* srcp = src->GetReadPtr(plane);
    int w = src->GetRowSize(plane) / pixelsize;
    int h = src->GetHeight(plane);

    if (w == 0 || h == 0)
      env->ThrowError("MinMax: plane does not exist!");

    int real_buffersize;
    std::unique_ptr<int[]> accum_buf;

    if (IS_CUDA) {
      // CUDA
      int bits_per_pixel = vi.BitsPerComponent();
      int buffersize = real_buffersize = (1 << min(16, bits_per_pixel));
      int pitch = src->GetPitch(plane) / pixelsize;

      VideoInfo workvi = VideoInfo();
      workvi.pixel_type = VideoInfo::CS_BGR32;
      workvi.width = 256;
      workvi.height = nblocks(buffersize * sizeof(int), workvi.width * 4);
      PVideoFrame work = env->NewVideoFrame(workvi);
      int* workbuf = reinterpret_cast<int*>(work->GetWritePtr());
      accum_buf = std::unique_ptr<int[]>(new int[buffersize]);


      if (w % 4)
        env->ThrowError("MinMax: width must be multiple of 4 on CUDA");

      int maxv = ((1 << bits_per_pixel) - 1);

      // Count each component
      switch (pixelsize) {
      case 1:
        calc_count_hist<uchar4>(srcp, w, h, pitch, maxv, workbuf, accum_buf.get(), buffersize, env);
        break;
      case 2:
        calc_count_hist<ushort4>(srcp, w, h, pitch, maxv, workbuf, accum_buf.get(), buffersize, env);
        break;
      case 4:
        calc_count_hist<float4>(srcp, w, h, pitch, maxv, workbuf, accum_buf.get(), buffersize, env);
        break;
      }
    }
    else {
      int pitch = src->GetPitch(plane);

      // CPU  int pixelsize = vi.ComponentSize();
      int buffersize = pixelsize == 1 ? 256 : 65536; // 65536 for float, too, reason for 10-14 bits: avoid overflow
      real_buffersize = pixelsize == 4 ? 65536 : (1 << vi.BitsPerComponent());
      accum_buf = std::unique_ptr<int[]>(new int[buffersize]);

      // Reset accumulators
      std::fill_n(accum_buf.get(), buffersize, 0);

      // Count each component
      if (pixelsize == 1) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            accum_buf[srcp[x]]++;
          }
          srcp += pitch;
        }
      }
      else if (pixelsize == 2) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            accum_buf[reinterpret_cast<const uint16_t *>(srcp)[x]]++;
          }
          srcp += pitch;
        }
      }
      else { //pixelsize==4 float
             // for float results are always checked with 16 bit precision only
             // or else we cannot populate non-digital steps with this standard method
             // See similar in colors, ColorYUV analyze
        const bool chroma = (plane == PLANAR_U) || (plane == PLANAR_V);
        if (chroma) {
#ifdef FLOAT_CHROMA_IS_ZERO_CENTERED
          const float shift = 32768.0f;
#else
          const float shift = 0.0f;
#endif
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              // -0.5..0.5 to 0..65535 when FLOAT_CHROMA_IS_ZERO_CENTERED
              const float pixel = reinterpret_cast<const float *>(srcp)[x];
              accum_buf[clamp((int)(65535.0f*pixel + shift + 0.5f), 0, 65535)]++;
            }
            srcp += pitch;
          }
        }
        else {
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              const float pixel = reinterpret_cast<const float *>(srcp)[x];
              accum_buf[clamp((int)(65535.0f * pixel + 0.5f), 0, 65535)]++;
            }
            srcp += pitch;
          }
        }
      }
    }

    int pixels = w * h;
    threshold /= 100.0;  // Thresh now 0-1
    threshold = clamp(threshold, 0.0, 1.0);

    unsigned int tpixels = (unsigned int)(pixels*threshold);

    int retval;

    // Find the value we need.
    if (mode == MIN) {
      unsigned int counted = 0;
      retval = real_buffersize - 1;
      for (int i = 0; i < real_buffersize; i++) {
        counted += accum_buf[i];
        if (counted > tpixels) {
          retval = i;
          break;
        }
      }
    }
    else if (mode == MAX) {
      unsigned int counted = 0;
      retval = 0;
      for (int i = real_buffersize - 1; i >= 0; i--) {
        counted += accum_buf[i];
        if (counted > tpixels) {
          retval = i;
          break;
        }
      }
    }
    else if (mode == MINMAX_DIFFERENCE) {
      unsigned int counted = 0;
      int i, t_min = 0;
      // Find min
      for (i = 0; i < real_buffersize; i++) {
        counted += accum_buf[i];
        if (counted > tpixels) {
          t_min = i;
          break;
        }
      }

      // Find max
      counted = 0;
      int t_max = real_buffersize - 1;
      for (i = real_buffersize - 1; i >= 0; i--) {
        counted += accum_buf[i];
        if (counted > tpixels) {
          t_max = i;
          break;
        }
      }

      retval = t_max - t_min; // results <0 will be returned if threshold > 50
    }
    else {
      retval = -1;
    }

    if (pixelsize == 4) {
      const bool chroma = (plane == PLANAR_U) || (plane == PLANAR_V);
      if (chroma && (mode == MIN && mode == MAX)) {
#ifdef FLOAT_CHROMA_IS_ZERO_CENTERED
        const float shift = 32768.0f;
#else
        const float shift = 0.0f;
#endif
        return AVSValue((double)(retval - shift) / (real_buffersize - 1)); // convert back to float, /65535
      }
      else {
        return AVSValue((double)retval / (real_buffersize - 1)); // convert back to float, /65535
      }
    }
    else
      return AVSValue(retval);
  }

  static AVSValue Create_max(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MAX, env);
  }
  static AVSValue Create_min(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MIN, env);
  }
  static AVSValue Create_median(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return MinMax(args[0], user_data, 50.0, args[1].AsInt(0), plane, MIN, env);
  }
  static AVSValue Create_minmax(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MINMAX_DIFFERENCE, env);
  }

private:
  enum { MIN = 1, MAX = 2, MEDIAN = 3, MINMAX_DIFFERENCE = 4 };

};
#pragma endregion

extern const FuncDefinition conditonal_functions[] = {
  {  "AverageLuma",    BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_Y },
  {  "AverageChromaU", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_U },
  {  "AverageChromaV", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_V },
  {  "AverageR", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_R },
  {  "AverageG", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_G },
  {  "AverageB", BUILTIN_FUNC_PREFIX, "c[offset]i", AveragePlane::Create, (void *)PLANAR_B },

  {  "RGBDifference",     BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)-1 },
  {  "LumaDifference",    BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_Y },
  {  "ChromaUDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_U },
  {  "ChromaVDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_V },
  {  "RDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_R },
  {  "GDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_G },
  {  "BDifference", BUILTIN_FUNC_PREFIX, "cc", ComparePlane::Create, (void *)PLANAR_B },

  {  "YDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_Y },
  {  "UDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_U },
  {  "VDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_V },
  {  "RGBDifferenceFromPrevious", BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)-1 },
  {  "RDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_R },
  {  "GDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_G },
  {  "BDifferenceFromPrevious",   BUILTIN_FUNC_PREFIX, "c", ComparePlane::Create_prev, (void *)PLANAR_B },

  {  "YDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_Y },
  {  "UDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_U },
  {  "VDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_V },
  {  "RGBDifferenceToNext", BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)-1 },
  {  "RDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_R },
  {  "GDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_G },
  {  "BDifferenceToNext",   BUILTIN_FUNC_PREFIX, "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_B },

  {  "YPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_Y },
  {  "YPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_Y },
  {  "YPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_Y },
  {  "UPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_U },
  {  "UPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_U },
  {  "UPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_U },
  {  "VPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "VPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "VPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_V },
  {  "RPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_R },
  {  "RPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_R },
  {  "RPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_R },
  {  "GPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_G },
  {  "GPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_G },
  {  "GPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_G },
  {  "BPlaneMax",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_B },
  {  "BPlaneMin",    BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_B },
  {  "BPlaneMedian", BUILTIN_FUNC_PREFIX, "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_B },
  {  "YPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_Y },
  {  "UPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_U }, // AVS+! was before: missing offset parameter
  {  "VPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_V }, // AVS+! was before: missing offset parameter
  {  "RPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_R },
  {  "GPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_G },
  {  "BPlaneMinMaxDifference", BUILTIN_FUNC_PREFIX, "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_B },

  { 0 }
};
