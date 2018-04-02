
#include <limits>
#include <memory>

#include "AvsCUDA.h"

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
  int tid = threadIdx.x + threadIdx.y;

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
float calc_sum_of_pixels(const void* src, int width, int height, int pitch, int maxv, void* sum, IScriptEnvironment* env)
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
  return (float)(ret / (height * width));;
}

class AveragePlane {

public:
  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case PLANAR_Y:
        return env->Invoke("AverageLuma", args);
      case PLANAR_U:
        return env->Invoke("AverageChromaU", args);
      case PLANAR_V:
        return env->Invoke("AverageChromaV", args);
      case PLANAR_R:
        return env->Invoke("AverageR", args);
      case PLANAR_G:
        return env->Invoke("AverageG", args);
      case PLANAR_B:
        return env->Invoke("AverageB", args);
      default:
        assert(0);
        break;
      }
    }
    return AvgPlane(args[0], user_data, plane, args[1].AsInt(0), env);
  }
  static AVSValue AvgPlane(AVSValue clip, void* user_data, int plane, int offset, IScriptEnvironment* env)
  {
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
    int pitch = src->GetPitch(plane) / pixelsize;

    if (width == 0 || height == 0)
      env->ThrowError("Average Plane: plane does not exist!");

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
      if(sum_in_32bits)
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

    return (AVSValue)0;
  }
};

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
  int tid = threadIdx.x + threadIdx.y;

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
float calc_sad(const void* src0, const void* src1, int width, int height, int pitch, int maxv, void* sum, bool is_rgb, IScriptEnvironment* env)
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
    return (float)((sad * 4) / (height * width * 3)); // why * 4/3? alpha plane was masked out, anyway
  else
    return (float)(sad / (height * width));
}

class ComparePlane {

public:
  static AVSValue CmpPlane(AVSValue clip, AVSValue clip2, void* user_data, int plane, IScriptEnvironment* env)
  {
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
    const int pitch = src->GetPitch(plane) / pixelsize;;
    const int height2 = src2->GetHeight(plane);
    const int rowsize2 = src2->GetRowSize(plane);
    const int width2 = rowsize2 / pixelsize;
    const int pitch2 = src2->GetPitch(plane) / pixelsize;;

    if (vi.ComponentSize() != vi2.ComponentSize())
      env->ThrowError("Plane Difference: Bit-depth are not the same!");

    if (width == 0 || height == 0)
      env->ThrowError("Plane Difference: plane does not exist!");

    if (width % 4)
      env->ThrowError("Plane Difference: width must be multiple of 4 on CUDA");

    if (height != height2 || width != width2)
      env->ThrowError("Plane Difference: Images are not the same size!");

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
  static AVSValue CmpPlaneSame(AVSValue clip, void* user_data, int offset, int plane, IScriptEnvironment* env)
  {
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
    int pitch = src->GetPitch(plane) / pixelsize;;
    int pitch2 = src2->GetPitch(plane) / pixelsize;;

    if (width == 0 || height == 0)
      env->ThrowError("Plane Difference: No chroma planes in greyscale clip!");

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

  static AVSValue Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case -1:
        return env->Invoke("RGBDifference", args);
      case PLANAR_Y:
        return env->Invoke("LumaDifference", args);
      case PLANAR_U:
        return env->Invoke("ChromaUDifference", args);
      case PLANAR_V:
        return env->Invoke("ChromaVDifference", args);
      case PLANAR_R:
        return env->Invoke("RDifference", args);
      case PLANAR_G:
        return env->Invoke("GDifference", args);
      case PLANAR_B:
        return env->Invoke("BDifference", args);
      default:
        assert(0);
        break;
      }
    }
    return CmpPlane(args[0], args[1], user_data, plane, env);
  }
  static AVSValue Create_prev(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case -1:
        return env->Invoke("RGBDifferenceFromPrevious", args);
      case PLANAR_Y:
        return env->Invoke("YDifferenceFromPrevious", args);
      case PLANAR_U:
        return env->Invoke("UDifferenceFromPrevious", args);
      case PLANAR_V:
        return env->Invoke("VDifferenceFromPrevious", args);
      case PLANAR_R:
        return env->Invoke("RDifferenceFromPrevious", args);
      case PLANAR_G:
        return env->Invoke("GDifferenceFromPrevious", args);
      case PLANAR_B:
        return env->Invoke("BDifferenceFromPrevious", args);
      default:
        assert(0);
        break;
      }
    }
    return CmpPlaneSame(args[0], user_data, -1, plane, env);
  }
  static AVSValue Create_next(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case -1:
        return env->Invoke("RGBDifferenceToNext", args);
      case PLANAR_Y:
        return env->Invoke("YDifferenceToNext", args);
      case PLANAR_U:
        return env->Invoke("UDifferenceToNext", args);
      case PLANAR_V:
        return env->Invoke("VDifferenceToNext", args);
      case PLANAR_R:
        return env->Invoke("RDifferenceToNext", args);
      case PLANAR_G:
        return env->Invoke("GDifferenceToNext", args);
      case PLANAR_B:
        return env->Invoke("BDifferenceToNext", args);
      default:
        assert(0);
        break;
      }
    }
    return CmpPlaneSame(args[0], user_data, args[1].AsInt(1), plane, env);
  }
};


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
__device__ int4 to_count_index(float4 src, int maxv) { return to_int(clamp(src * 65536.0f, 0.0f, 65536.0f)); }

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
  IScriptEnvironment* env)
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

class MinMaxPlane {

public:
  static AVSValue MinMax(AVSValue clip, void* user_data, double threshold, int offset, int plane, int mode, IScriptEnvironment* env)
  {
    if (!clip.IsClip())
      env->ThrowError("MinMax: No clip supplied!");

    PClip child = clip.AsClip();
    VideoInfo vi = child->GetVideoInfo();

    if (!vi.IsPlanar())
      env->ThrowError("MinMax: Image must be planar");

    int bits_per_pixel = vi.BitsPerComponent();
    int buffersize = (1 << min(16, bits_per_pixel));

    VideoInfo workvi = VideoInfo();
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 256;
    workvi.height = nblocks(buffersize * sizeof(int), workvi.width * 4);
    PVideoFrame work = env->NewVideoFrame(workvi);
    int* workbuf = reinterpret_cast<int*>(work->GetWritePtr());
    std::unique_ptr<int[]> accum_buf = std::unique_ptr<int[]>(new int[buffersize]);

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
    int pitch = src->GetPitch(plane) / pixelsize;
    int w = src->GetRowSize(plane) / pixelsize;
    int h = src->GetHeight(plane);

    if (w == 0 || h == 0)
      env->ThrowError("MinMax: plane does not exist!");

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

    int pixels = w * h;
    threshold /= 100.0;  // Thresh now 0-1
    threshold = clamp(threshold, 0.0, 1.0);

    unsigned int tpixels = (unsigned int)(pixels*threshold);

    int real_buffersize = buffersize;
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

    if (pixelsize == 4)
      return AVSValue((double)retval / (real_buffersize - 1)); // convert back to float, /65535
    else
      return AVSValue(retval);
  }

  static AVSValue Create_max(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case PLANAR_Y:
        return env->Invoke("YPlaneMax", args);
      case PLANAR_U:
        return env->Invoke("UPlaneMax", args);
      case PLANAR_V:
        return env->Invoke("VPlaneMax", args);
      case PLANAR_R:
        return env->Invoke("RPlaneMax", args);
      case PLANAR_G:
        return env->Invoke("GPlaneMax", args);
      case PLANAR_B:
        return env->Invoke("BPlaneMax", args);
      default:
        assert(0);
        break;
      }
    }
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MAX, env);
  }
  static AVSValue Create_min(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case PLANAR_Y:
        return env->Invoke("YPlaneMin", args);
      case PLANAR_U:
        return env->Invoke("UPlaneMin", args);
      case PLANAR_V:
        return env->Invoke("VPlaneMin", args);
      case PLANAR_R:
        return env->Invoke("RPlaneMin", args);
      case PLANAR_G:
        return env->Invoke("GPlaneMin", args);
      case PLANAR_B:
        return env->Invoke("BPlaneMin", args);
      default:
        assert(0);
        break;
      }
    }
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MIN, env);
  }
  static AVSValue Create_median(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case PLANAR_Y:
        return env->Invoke("YPlaneMedian", args);
      case PLANAR_U:
        return env->Invoke("UPlaneMedian", args);
      case PLANAR_V:
        return env->Invoke("VPlaneMedian", args);
      case PLANAR_R:
        return env->Invoke("RPlaneMedian", args);
      case PLANAR_G:
        return env->Invoke("GPlaneMedian", args);
      case PLANAR_B:
        return env->Invoke("BPlaneMedian", args);
      default:
        assert(0);
        break;
      }
    }
    return MinMax(args[0], user_data, 50.0, args[1].AsInt(0), plane, MIN, env);
  }
  static AVSValue Create_minmax(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    int plane = (int)reinterpret_cast<intptr_t>(user_data);
    if (!IS_CUDA) {
      // CUDAじゃない場合はAvisynth組み込み関数を呼び出す
      switch (plane) {
      case PLANAR_Y:
        return env->Invoke("YPlaneMinMaxDifference", args);
      case PLANAR_U:
        return env->Invoke("UPlaneMinMaxDifference", args);
      case PLANAR_V:
        return env->Invoke("VPlaneMinMaxDifference", args);
      case PLANAR_R:
        return env->Invoke("RPlaneMinMaxDifference", args);
      case PLANAR_G:
        return env->Invoke("GPlaneMinMaxDifference", args);
      case PLANAR_B:
        return env->Invoke("BPlaneMinMaxDifference", args);
      default:
        assert(0);
        break;
      }
    }
    return MinMax(args[0], user_data, args[1].AsDblDef(0.0), args[2].AsInt(0), plane, MINMAX_DIFFERENCE, env);
  }

private:
  enum { MIN = 1, MAX = 2, MEDIAN = 3, MINMAX_DIFFERENCE = 4 };

};

extern const FuncDefinition conditonal_functions[] = {
  { "KAverageLuma",    "c[offset]i", AveragePlane::Create, (void *)PLANAR_Y },
  { "KAverageChromaU", "c[offset]i", AveragePlane::Create, (void *)PLANAR_U },
  { "KAverageChromaV", "c[offset]i", AveragePlane::Create, (void *)PLANAR_V },
  { "KAverageR", "c[offset]i", AveragePlane::Create, (void *)PLANAR_R },
  { "KAverageG", "c[offset]i", AveragePlane::Create, (void *)PLANAR_G },
  { "KAverageB", "c[offset]i", AveragePlane::Create, (void *)PLANAR_B },

  { "KRGBDifference",     "cc", ComparePlane::Create, (void *)-1 },
  { "KLumaDifference",    "cc", ComparePlane::Create, (void *)PLANAR_Y },
  { "KChromaUDifference", "cc", ComparePlane::Create, (void *)PLANAR_U },
  { "KChromaVDifference", "cc", ComparePlane::Create, (void *)PLANAR_V },
  { "KRDifference", "cc", ComparePlane::Create, (void *)PLANAR_R },
  { "KGDifference", "cc", ComparePlane::Create, (void *)PLANAR_G },
  { "KBDifference", "cc", ComparePlane::Create, (void *)PLANAR_B },

  { "KYDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_Y },
  { "KUDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_U },
  { "KVDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_V },
  { "KRGBDifferenceFromPrevious", "c", ComparePlane::Create_prev, (void *)-1 },
  { "KRDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_R },
  { "KGDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_G },
  { "KBDifferenceFromPrevious",   "c", ComparePlane::Create_prev, (void *)PLANAR_B },

  { "KYDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_Y },
  { "KUDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_U },
  { "KVDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_V },
  { "KRGBDifferenceToNext", "c[offset]i", ComparePlane::Create_next, (void *)-1 },
  { "KRDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_R },
  { "KGDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_G },
  { "KBDifferenceToNext",   "c[offset]i", ComparePlane::Create_next, (void *)PLANAR_B },

  { "KYPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_Y },
  { "KYPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_Y },
  { "KYPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_Y },
  { "KUPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_U },
  { "KUPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_U },
  { "KUPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_U },
  { "KVPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_V },
  { "KVPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_V },
  { "KVPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_V },
  { "KRPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_R },
  { "KRPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_R },
  { "KRPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_R },
  { "KGPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_G },
  { "KGPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_G },
  { "KGPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_G },
  { "KBPlaneMax",    "c[threshold]f[offset]i", MinMaxPlane::Create_max, (void *)PLANAR_B },
  { "KBPlaneMin",    "c[threshold]f[offset]i", MinMaxPlane::Create_min, (void *)PLANAR_B },
  { "KBPlaneMedian", "c[offset]i", MinMaxPlane::Create_median, (void *)PLANAR_B },

  { "KYPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_Y },
  { "KUPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_U },
  { "KVPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_V },
  { "KRPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_R },
  { "KGPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_G },
  { "KBPlaneMinMaxDifference", "c[threshold]f[offset]i", MinMaxPlane::Create_minmax, (void *)PLANAR_B },

  { 0 }
};
