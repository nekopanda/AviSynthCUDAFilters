
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "CommonFunctions.h"
#include "KFM.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"
#include "TextOut.h"

template <typename vpixel_t>
void cpu_calc_field_diff(const vpixel_t* ptr, int nt, int width, int height, int pitch, unsigned long long int *sum)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 combe = CalcCombe(
        to_int(ptr[x + (y - 2) * pitch]),
        to_int(ptr[x + (y - 1) * pitch]),
        to_int(ptr[x + (y + 0) * pitch]),
        to_int(ptr[x + (y + 1) * pitch]),
        to_int(ptr[x + (y + 2) * pitch]));

      *sum += ((combe.x > nt) ? combe.x : 0);
      *sum += ((combe.y > nt) ? combe.y : 0);
      *sum += ((combe.z > nt) ? combe.z : 0);
      *sum += ((combe.w > nt) ? combe.w : 0);
    }
  }
}

enum {
  CALC_FIELD_DIFF_X = 32,
  CALC_FIELD_DIFF_Y = 16,
  CALC_FIELD_DIFF_THREADS = CALC_FIELD_DIFF_X * CALC_FIELD_DIFF_Y
};

__global__ void kl_init_uint64(uint64_t* sum)
{
  sum[threadIdx.x] = 0;
}

template <typename vpixel_t>
__global__ void kl_calculate_field_diff(const vpixel_t* ptr, int nt, int width, int height, int pitch, uint64_t* sum)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * CALC_FIELD_DIFF_X;

  int tmpsum = 0;
  if (x < width && y < height) {
    int4 combe = CalcCombe(
      to_int(ptr[x + (y - 2) * pitch]),
      to_int(ptr[x + (y - 1) * pitch]),
      to_int(ptr[x + (y + 0) * pitch]),
      to_int(ptr[x + (y + 1) * pitch]),
      to_int(ptr[x + (y + 2) * pitch]));

    tmpsum += ((combe.x > nt) ? combe.x : 0);
    tmpsum += ((combe.y > nt) ? combe.y : 0);
    tmpsum += ((combe.z > nt) ? combe.z : 0);
    tmpsum += ((combe.w > nt) ? combe.w : 0);
  }

  __shared__ int sbuf[CALC_FIELD_DIFF_THREADS];
  dev_reduce<int, CALC_FIELD_DIFF_THREADS, AddReducer<int>>(tid, tmpsum, sbuf);

  if (tid == 0) {
    atomicAdd(sum, tmpsum);
  }
}

class KFieldDiff : public KFMFilterBase
{
  int nt6;
  bool chroma;

  VideoInfo padvi;
  VideoInfo workvi;

  template <typename pixel_t>
  unsigned long long int CalcFieldDiff(Frame& frame, Frame& work, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = frame.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* srcU = frame.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* srcV = frame.GetReadPtr<vpixel_t>(PLANAR_V);
    unsigned long long int* sum = work.GetWritePtr<unsigned long long int>();

    int pitchY = frame.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = frame.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_init_uint64 << <1, 1 >> > (sum);
      DEBUG_SYNC;
      kl_calculate_field_diff << <blocks, threads >> >(srcY, nt6, width4, vi.height, pitchY, sum);
      DEBUG_SYNC;
      if (chroma) {
        kl_calculate_field_diff << <blocksUV, threads >> > (srcU, nt6, width4UV, heightUV, pitchUV, sum);
        DEBUG_SYNC;
        kl_calculate_field_diff << <blocksUV, threads >> > (srcV, nt6, width4UV, heightUV, pitchUV, sum);
        DEBUG_SYNC;
      }
      long long int result;
      CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(*sum), cudaMemcpyDeviceToHost));
      return result;
    }
    else {
      *sum = 0;
      cpu_calc_field_diff(srcY, nt6, width4, vi.height, pitchY, sum);
      if (chroma) {
        cpu_calc_field_diff(srcU, nt6, width4UV, heightUV, pitchUV, sum);
        cpu_calc_field_diff(srcV, nt6, width4UV, heightUV, pitchUV, sum);
      }
      return *sum;
    }
  }

  template <typename pixel_t>
  double InternalFieldDiff(int n, PNeoEnv env)
  {
    Frame src = child->GetFrame(n, env);
    Frame padded = Frame(env->NewVideoFrame(padvi), VPAD);
    Frame work = env->NewVideoFrame(workvi);

    CopyFrame<pixel_t>(src, padded, env);
    PadFrame<pixel_t>(padded, env);
    auto raw = CalcFieldDiff<pixel_t>(padded, work, env);
    raw /= 6; // 計算式から

    int shift = vi.BitsPerComponent() - 8; // 8bitに合わせる
    return (double)(raw >> shift);
  }

public:
  KFieldDiff(PClip clip, float nt, bool chroma)
    : KFMFilterBase(clip)
    , nt6(scaleParam(nt * 6, vi.BitsPerComponent()))
    , chroma(chroma)
    , padvi(vi)
  {
    padvi.height += VPAD * 2;

    int work_bytes = sizeof(long long int);
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 4;
    workvi.height = nblocks(work_bytes, workvi.width * 4);
  }

  AVSValue ConditionalFieldDiff(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return InternalFieldDiff<uint8_t>(n, env);
    case 2:
      return InternalFieldDiff<uint16_t>(n, env);
    default:
      env->ThrowError("[KFieldDiff] Unsupported pixel format");
    }

    return 0;
  }

  static AVSValue __cdecl CFunc(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    AVSValue cnt = env->GetVar("current_frame");
    if (!cnt.IsInt()) {
      env->ThrowError("[KCFieldDiff] This filter can only be used within ConditionalFilter!");
    }
    int n = cnt.AsInt();
    std::unique_ptr<KFieldDiff> f = std::unique_ptr<KFieldDiff>(new KFieldDiff(
      args[0].AsClip(),    // clip
      (float)args[1].AsFloat(3),    // nt
      args[2].AsBool(true) // chroma
    ));
    return f->ConditionalFieldDiff(n, env);
  }
};


void cpu_init_block_sum(int *sumAbs, int* sumSig, int* maxSum, int length)
{
  for (int x = 0; x < length; ++x) {
    sumAbs[x] = 0;
    sumSig[x] = 0;
  }
  maxSum[0] = 0;
}

template <typename vpixel_t, int BLOCK_SIZE, int TH_Z>
void cpu_add_block_sum(
  const vpixel_t* src0,
  const vpixel_t* src1,
  int width, int height, int pitch,
  int blocks_w, int blocks_h, int block_pitch,
  int *sumAbs, int* sumSig)
{
  for (int by = 0; by < blocks_h; ++by) {
    for (int bx = 0; bx < blocks_w; ++bx) {
      int abssum = 0;
      int sigsum = 0;
      for (int ty = 0; ty < BLOCK_SIZE; ++ty) {
        for (int tx = 0; tx < BLOCK_SIZE / 4; ++tx) {
          int x = tx + bx * BLOCK_SIZE / 4;
          int y = ty + by * BLOCK_SIZE;
          if (x < width && y < height) {
            auto s0 = src0[x + y * pitch];
            auto s1 = src1[x + y * pitch];
            auto t0 = absdiff(s0, s1);
            auto t1 = to_int(s0) - to_int(s1);
            abssum += t0.x + t0.y + t0.z + t0.w;
            sigsum += t1.x + t1.y + t1.z + t1.w;
          }
        }
      }
      sumAbs[bx + by * block_pitch] += abssum;
      sumSig[bx + by * block_pitch] += sigsum;
    }
  }
}

__global__ void kl_init_block_sum(int *sumAbs, int* sumSig, int* maxSum, int length)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if (x < length) {
    sumAbs[x] = 0;
    sumSig[x] = 0;
  }
  if (x == 0) {
    maxSum[0] = 0;
  }
}

template <typename vpixel_t, int BLOCK_SIZE, int TH_Z>
__global__ void kl_add_block_sum(
  const vpixel_t* __restrict__ src0,
  const vpixel_t* __restrict__ src1,
  int width, int height, int pitch,
  int blocks_w, int blocks_h, int block_pitch,
  int *sumAbs, int* sumSig)
{
  // blockDim.x == BLOCK_SIZE/4
  // blockDim.y == BLOCK_SIZE
  enum { PIXELS = BLOCK_SIZE * BLOCK_SIZE / 4 };
  int bx = threadIdx.z + TH_Z * blockIdx.x;
  int by = blockIdx.y;
  int x = threadIdx.x + (BLOCK_SIZE / 4) * bx;
  int y = threadIdx.y + BLOCK_SIZE * by;
  int tz = threadIdx.z;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  int abssum = 0;
  int sigsum = 0;
  if (x < width && y < height) {
    auto s0 = src0[x + y * pitch];
    auto s1 = src1[x + y * pitch];
    auto t0 = absdiff(s0, s1);
    auto t1 = to_int(s0) - to_int(s1);
    abssum = t0.x + t0.y + t0.z + t0.w;
    sigsum = t1.x + t1.y + t1.z + t1.w;
  }

  __shared__ int sbuf[TH_Z][PIXELS];
  dev_reduce<int, PIXELS, AddReducer<int>>(tid, abssum, sbuf[tz]);
  dev_reduce<int, PIXELS, AddReducer<int>>(tid, sigsum, sbuf[tz]);

  if (tid == 0) {
    sumAbs[bx + by * block_pitch] += abssum;
    sumSig[bx + by * block_pitch] += sigsum;
  }
}

template <typename vpixel_t, int BLOCK_SIZE, int TH_Z>
void launch_add_block_sum(
  const vpixel_t* src0,
  const vpixel_t* src1,
  int width, int height, int pitch,
  int blocks_w, int blocks_h, int block_pitch,
  int *sumAbs, int* sumSig)
{
  dim3 threads(BLOCK_SIZE >> 2, BLOCK_SIZE, TH_Z);
  dim3 blocks(nblocks(blocks_w, TH_Z), blocks_h);
  kl_add_block_sum<vpixel_t, BLOCK_SIZE, TH_Z> << <blocks, threads >> >(src0, src1,
    width, height, pitch, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
}

void cpu_block_sum_max(
  const int4* sumAbs, const int4* sumSig,
  int blocks_w, int blocks_h, int block_pitch,
  int* highest_sum)
{
  int tmpmax = 0;
  for (int y = 0; y < blocks_h; ++y) {
    for (int x = 0; x < blocks_w; ++x) {
      int4 metric = sumAbs[x + y * block_pitch] + sumSig[x + y * block_pitch] * 4;
      tmpmax = max(tmpmax, max(max(metric.x, metric.y), max(metric.z, metric.w)));
    }
  }
  *highest_sum = tmpmax;
}

__global__ void kl_block_sum_max(
  const int4* sumAbs, const int4* sumSig,
  int blocks_w, int blocks_h, int block_pitch,
  int* highest_sum)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * CALC_FIELD_DIFF_X;

  int tmpmax = 0;
  if (x < blocks_w && y < blocks_h) {
    int4 metric = sumAbs[x + y * block_pitch] + sumSig[x + y * block_pitch] * 4;
    tmpmax = max(max(metric.x, metric.y), max(metric.z, metric.w));
  }

  __shared__ int sbuf[CALC_FIELD_DIFF_THREADS];
  dev_reduce<int, CALC_FIELD_DIFF_THREADS, MaxReducer<int>>(tid, tmpmax, sbuf);

  if (tid == 0) {
    atomicMax(highest_sum, tmpmax);
  }
}

class KFrameDiffDup : public KFMFilterBase
{
  bool chroma;
  int blocksize;

  int logUVx;
  int logUVy;

  int th_z, th_uv_z;
  int blocks_w, blocks_h, block_pitch;
  VideoInfo workvi;

  enum { THREADS = 256 };

  // returns argmax(subAbs + sumSig * 4)
  template <typename pixel_t>
  int CalcFrameDiff(Frame& src0, Frame& src1, Frame& work, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* src0Y = src0.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src0U = src0.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src0V = src0.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* src1Y = src1.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src1U = src1.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src1V = src1.GetReadPtr<vpixel_t>(PLANAR_V);
    int* sumAbs = work.GetWritePtr<int>();
    int* sumSig = &sumAbs[block_pitch * blocks_h];
    int* maxSum = &sumSig[block_pitch * blocks_h];

    int pitchY = src0.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = src0.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;
    int blocks_w4 = blocks_w >> 2;
    int block_pitch4 = block_pitch >> 2;

    void(*table[2][4])(
      const vpixel_t* src0,
      const vpixel_t* src1,
      int width, int height, int pitch,
      int blocks_w, int blocks_h, int block_pitch,
      int *sumAbs, int* sumSig) =
    {
      {
        launch_add_block_sum<vpixel_t, 32, THREADS / (32 * (32 / 4))>,
        launch_add_block_sum<vpixel_t, 16, THREADS / (16 * (16 / 4))>,
        launch_add_block_sum<vpixel_t, 8, THREADS / (8 * (8 / 4))>,
        launch_add_block_sum<vpixel_t, 4, THREADS / (4 * (4 / 4))>,
      },
      {
        cpu_add_block_sum<vpixel_t, 32, THREADS / (32 * (32 / 4))>,
        cpu_add_block_sum<vpixel_t, 16, THREADS / (16 * (16 / 4))>,
        cpu_add_block_sum<vpixel_t, 8, THREADS / (8 * (8 / 4))>,
        cpu_add_block_sum<vpixel_t, 4, THREADS / (4 * (4 / 4))>,
      }
    };

    int f_idx;
    switch (blocksize) {
    case 32: f_idx = 0; break;
    case 16: f_idx = 1; break;
    case 8: f_idx = 2; break;
    }

    if (IS_CUDA) {
      kl_init_block_sum << <64, nblocks(block_pitch * blocks_h, 64) >> > (
        sumAbs, sumSig, maxSum, block_pitch * blocks_h);
      DEBUG_SYNC;
      table[0][f_idx](src0Y, src1Y,
        width4, vi.height, pitchY, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
      DEBUG_SYNC;
      if (chroma) {
        table[0][f_idx + logUVx](src0U, src1U,
          width4UV, heightUV, pitchUV, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
        DEBUG_SYNC;
        table[0][f_idx + logUVx](src0V, src1V,
          width4UV, heightUV, pitchUV, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
        DEBUG_SYNC;
      }
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(blocks_w4, threads.x), nblocks(blocks_h, threads.y));
      kl_block_sum_max << <blocks, threads >> > (
        (int4*)sumAbs, (int4*)sumSig, blocks_w4, blocks_h, block_pitch4, maxSum);
      int result;
      CUDA_CHECK(cudaMemcpy(&result, maxSum, sizeof(int), cudaMemcpyDeviceToHost));
      return result;
    }
    else {
      cpu_init_block_sum(
        sumAbs, sumSig, maxSum, block_pitch * blocks_h);
      table[1][f_idx](src0Y, src1Y,
        width4, vi.height, pitchY, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
      if (chroma) {
        table[1][f_idx + logUVx](src0U, src1U,
          width4UV, heightUV, pitchUV, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
        table[1][f_idx + logUVx](src0V, src1V,
          width4UV, heightUV, pitchUV, blocks_w, blocks_h, block_pitch, sumAbs, sumSig);
      }
      cpu_block_sum_max(
        (int4*)sumAbs, (int4*)sumSig, blocks_w4, blocks_h, block_pitch4, maxSum);
      return *maxSum;
    }
  }

  template <typename pixel_t>
  double InternalFrameDiff(int n, PNeoEnv env)
  {
    Frame src0 = child->GetFrame(clamp(n - 1, 0, vi.num_frames - 1), env);
    Frame src1 = child->GetFrame(clamp(n, 0, vi.num_frames - 1), env);
    Frame work = env->NewVideoFrame(workvi);

    int diff = CalcFrameDiff<pixel_t>(src0, src1, work, env);

    int shift = vi.BitsPerComponent() - 8;

    // dup232aだとこうだけど、この計算式はおかしいと思うので修正
    //return  diff / (64.0 * (235 << shift) * blocksize) * 100.0;
    return  diff / (2.0 * (235 << shift) * blocksize * blocksize) * 100.0;
  }

public:
  KFrameDiffDup(PClip clip, bool chroma, int blocksize)
    : KFMFilterBase(clip)
    , chroma(chroma)
    , blocksize(blocksize)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    blocks_w = nblocks(vi.width, blocksize);
    blocks_h = nblocks(vi.height, blocksize);

    th_z = THREADS / (blocksize * (blocksize / 4));
    th_uv_z = th_z * (chroma ? (1 << (logUVx + logUVy)) : 1);

    int block_align = max(4, th_uv_z);
    block_pitch = nblocks(blocks_w, block_align) * block_align;

    int work_bytes = sizeof(int) * block_pitch * blocks_h * 2 + sizeof(int);
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 256;
    workvi.height = nblocks(work_bytes, workvi.width * 4);
  }

  AVSValue ConditionalFrameDiff(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return InternalFrameDiff<uint8_t>(n, env);
    case 2:
      return InternalFrameDiff<uint16_t>(n, env);
    default:
      env->ThrowError("[KFrameDiffDup] Unsupported pixel format");
    }

    return 0;
  }

  static AVSValue __cdecl CFunc(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    AVSValue cnt = env->GetVar("current_frame");
    if (!cnt.IsInt()) {
      env->ThrowError("[KFrameDiffDup] This filter can only be used within ConditionalFilter!");
    }
    int n = cnt.AsInt();
    std::unique_ptr<KFrameDiffDup> f = std::unique_ptr<KFrameDiffDup>(new KFrameDiffDup(
      args[0].AsClip(),     // clip
      args[1].AsBool(true), // chroma
      args[2].AsInt(32)     // blocksize
    ));
    return f->ConditionalFrameDiff(n, env);
  }
};

__host__ __device__ int dev_limitter(int x, int nmin, int range) {
  return (x == 128)
    ? 128 
    : ((x < 128)
      ? ((((127 - range) < x)&(x < (128 - nmin))) ? 0 : 56)
      : ((((128 + nmin) < x)&(x < (129 + range))) ? 255 : 199));
}

void cpu_noise_clip(uchar4* dst, const uchar4* src, const uchar4* noise,
  int width, int height, int pitch, int nmin, int range)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto s = (to_int(src[x + y * pitch]) - to_int(noise[x + y * pitch]) + 256) >> 1;
      int4 tmp = {
        dev_limitter(s.x, nmin, range),
        dev_limitter(s.y, nmin, range),
        dev_limitter(s.z, nmin, range),
        dev_limitter(s.w, nmin, range)
      };
      dst[x + y * pitch] = VHelper<uchar4>::cast_to(tmp);
    }
  }
}

__global__ void kl_noise_clip(uchar4* dst, const uchar4* src, const uchar4* noise,
  int width, int height, int pitch, int nmin, int range)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto s = (to_int(src[x + y * pitch]) - to_int(noise[x + y * pitch]) + 256) >> 1;
    int4 tmp = {
      dev_limitter(s.x, nmin, range),
      dev_limitter(s.y, nmin, range),
      dev_limitter(s.z, nmin, range),
      dev_limitter(s.w, nmin, range)
    };
    dst[x + y * pitch] = VHelper<uchar4>::cast_to(tmp);
  }
}

class KNoiseClip : public KFMFilterBase
{
  PClip noiseclip;

  int range_y;
  int range_uv;
  int nmin_y;
  int nmin_uv;

  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    Frame src = child->GetFrame(n, env);
    Frame noise = noiseclip->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

    const vpixel_t* srcY = src.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* srcU = src.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* srcV = src.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* noiseY = noise.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* noiseU = noise.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* noiseV = noise.GetReadPtr<vpixel_t>(PLANAR_V);
    vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
    vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchY = src.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = src.GetPitch<vpixel_t>(PLANAR_U);
    int width = src.GetWidth<vpixel_t>(PLANAR_Y);
    int widthUV = src.GetWidth<vpixel_t>(PLANAR_U);
    int height = src.GetHeight(PLANAR_Y);
    int heightUV = src.GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_noise_clip << <blocks, threads >> >(dstY, srcY, noiseY, width, height, pitchY, nmin_y, range_y);
      DEBUG_SYNC;
      kl_noise_clip << <blocksUV, threads >> >(dstU, srcU, noiseU, widthUV, heightUV, pitchUV, nmin_uv, range_uv);
      DEBUG_SYNC;
      kl_noise_clip << <blocksUV, threads >> >(dstV, srcV, noiseV, widthUV, heightUV, pitchUV, nmin_uv, range_uv);
      DEBUG_SYNC;
    }
    else {
      cpu_noise_clip(dstY, srcY, noiseY, width, height, pitchY, nmin_y, range_y);
      cpu_noise_clip(dstU, srcU, noiseU, widthUV, heightUV, pitchUV, nmin_uv, range_uv);
      cpu_noise_clip(dstV, srcV, noiseV, widthUV, heightUV, pitchUV, nmin_uv, range_uv);
    }

    return dst.frame;
  }
public:
  KNoiseClip(PClip src, PClip noise,
    int nmin_y, int range_y, int nmin_uv, int range_uv, IScriptEnvironment* env)
    : KFMFilterBase(src)
    , noiseclip(noise)
    , range_y(range_y)
    , range_uv(range_uv)
    , nmin_y(nmin_y)
    , nmin_uv(nmin_uv)
  {
    VideoInfo noisevi = noiseclip->GetVideoInfo();

    if (vi.width & 3) env->ThrowError("[KNoiseClip]: width must be multiple of 4");
    if (vi.height & 3) env->ThrowError("[KNoiseClip]: height must be multiple of 4");
    if (vi.width != noisevi.width || vi.height != noisevi.height) {
      env->ThrowError("[KNoiseClip]: src and noiseclip must be same resoluction");
    }
    if (!(GetDeviceTypes(src) & GetDeviceTypes(noise))) {
      env->ThrowError("[KNoiseClip] Device unmatch. Two source must be same device.");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT(n, env);
    //case 2:
    //  dst = InternalGetFrame<uint16_t>(n60, fmframe, frameType, env);
    //  break;
    default:
      env->ThrowError("[KNoiseClip] Unsupported pixel format");
      break;
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KNoiseClip(
      args[0].AsClip(),       // src
      args[1].AsClip(),       // noise
      args[2].AsInt(1),       // nmin_y
      args[3].AsInt(128),     // range_y
      args[4].AsInt(1),       // nmin_uv
      args[5].AsInt(128),     // range_uv
      env
    );
  }
};


__host__ __device__ int dev_horizontal_sum(int4 s) {
  return s.x + s.y + s.z + s.w;
}

void cpu_analyze_noise(uint64_t* result,
  const uchar4* src0, const uchar4* src1, const uchar4* src2,
  int width, int height, int pitch)
{
  uint64_t sum0 = 0, sum1 = 0, sumR0 = 0, sumR1 = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto s0 = to_int(src0[x + y * pitch]);
      auto s1 = to_int(src1[x + y * pitch]);
      auto s2 = to_int(src2[x + y * pitch]);

      sum0 += dev_horizontal_sum(abs(s0 + (-128)));
      sum1 += dev_horizontal_sum(abs(s1 + (-128)));
      sumR0 += dev_horizontal_sum(abs(s1 - s0));
      sumR1 += dev_horizontal_sum(abs(s2 - s1));
    }
  }
  result[0] += sum0;
  result[1] += sum1;
  result[2] += sumR0;
  result[3] += sumR1;
}

__global__ void kl_analyze_noise(
  uint64_t* result,
  const uchar4* src0, const uchar4* src1, const uchar4* src2,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * CALC_FIELD_DIFF_X;

  int sum[4] = { 0 };
  if (x < width && y < height) {
    auto s0 = to_int(src0[x + y * pitch]);
    auto s1 = to_int(src1[x + y * pitch]);
    auto s2 = to_int(src2[x + y * pitch]);

    sum[0] = dev_horizontal_sum(abs(s0 + (-128)));
    sum[1] = dev_horizontal_sum(abs(s1 + (-128)));
    sum[2] = dev_horizontal_sum(abs(s1 - s0));
    sum[3] = dev_horizontal_sum(abs(s2 - s1));
  }

  __shared__ int sbuf[CALC_FIELD_DIFF_THREADS * 4];
  dev_reduceN<int, 4, CALC_FIELD_DIFF_THREADS, AddReducer<int>>(tid, sum, sbuf);

  if (tid == 0) {
    atomicAdd(&result[0], sum[0]);
    atomicAdd(&result[1], sum[1]);
    atomicAdd(&result[2], sum[2]);
    atomicAdd(&result[3], sum[3]);
  }
}

template <typename vpixel_t>
void cpu_analyze_diff(
  uint64_t* result, const vpixel_t* f0, const vpixel_t* f1,
  int width, int height, int pitch)
{
  uint64_t sum0 = 0, sum1 = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 a = to_int(f0[x + (y - 2) * pitch]);
      int4 b = to_int(f0[x + (y - 1) * pitch]);
      int4 c = to_int(f0[x + (y + 0) * pitch]);
      int4 d = to_int(f0[x + (y + 1) * pitch]);
      int4 e = to_int(f0[x + (y + 2) * pitch]);

      // 現在のフレーム(f0)
      sum0 += dev_horizontal_sum(CalcCombe(a, b, c, d, e));

      // TFF前提
      // 現在のフレームのボトムフィールド（奇数ライン）と次のフレームのトップフィールド（偶数ライン）
      if (y & 1) {
        // yは奇数ライン
        a = to_int(f0[x + (y - 2) * pitch]);
        b = to_int(f1[x + (y - 1) * pitch]);
        c = to_int(f0[x + (y + 0) * pitch]);
        d = to_int(f1[x + (y + 1) * pitch]);
        e = to_int(f0[x + (y + 2) * pitch]);
        sum1 += dev_horizontal_sum(CalcCombe(a, b, c, d, e));
      }
      else {
        // yは偶数ライン
        a = to_int(f1[x + (y - 2) * pitch]);
        b = to_int(f0[x + (y - 1) * pitch]);
        c = to_int(f1[x + (y + 0) * pitch]);
        d = to_int(f0[x + (y + 1) * pitch]);
        e = to_int(f1[x + (y + 2) * pitch]);
        sum1 += dev_horizontal_sum(CalcCombe(a, b, c, d, e));
      }
    }
  }
  result[0] += sum0;
  result[1] += sum1;
}

template <typename vpixel_t>
__global__ void kl_analyze_diff(
  uint64_t* result, const vpixel_t* f0, const vpixel_t* f1,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid = threadIdx.x + threadIdx.y * CALC_FIELD_DIFF_X;

  int sum[2] = { 0 };
  if (x < width && y < height) {
    int4 a = to_int(f0[x + (y - 2) * pitch]);
    int4 b = to_int(f0[x + (y - 1) * pitch]);
    int4 c = to_int(f0[x + (y + 0) * pitch]);
    int4 d = to_int(f0[x + (y + 1) * pitch]);
    int4 e = to_int(f0[x + (y + 2) * pitch]);

    // 現在のフレーム(f0)
    sum[0] = dev_horizontal_sum(CalcCombe(a, b, c, d, e));

    // TFF前提
    // 現在のフレームのボトムフィールド（奇数ライン）と次のフレームのトップフィールド（偶数ライン）
    if (y & 1) {
      // yは奇数ライン
      // ↓必要なくても読むのをやめるとレジスタ使用数が25->32に増える
      a = to_int(f0[x + (y - 2) * pitch]);
      b = to_int(f1[x + (y - 1) * pitch]);
      c = to_int(f0[x + (y + 0) * pitch]);
      d = to_int(f1[x + (y + 1) * pitch]);
      e = to_int(f0[x + (y + 2) * pitch]);
      // ↓この行をifの外に持っていくとレジスタ使用数が25->39に増える
      sum[1] = dev_horizontal_sum(CalcCombe(a, b, c, d, e));
    }
    else {
      // yは偶数ライン
      // ↓必要なくても読むのをやめるとレジスタ使用数が25->32に増える
      a = to_int(f1[x + (y - 2) * pitch]);
      b = to_int(f0[x + (y - 1) * pitch]);
      c = to_int(f1[x + (y + 0) * pitch]);
      d = to_int(f0[x + (y + 1) * pitch]);
      e = to_int(f1[x + (y + 2) * pitch]);
      // ↓この行をifの外に持っていくとレジスタ使用数が25->39に増える
      sum[1] = dev_horizontal_sum(CalcCombe(a, b, c, d, e));
    }
  }

  __shared__ int sbuf[CALC_FIELD_DIFF_THREADS * 2];
  dev_reduceN<int, 2, CALC_FIELD_DIFF_THREADS, AddReducer<int>>(tid, sum, sbuf);

  if (tid == 0) {
    atomicAdd(&result[0], sum[0]);
    atomicAdd(&result[1], sum[1]);
  }
}

template __global__ void kl_analyze_diff(
  uint64_t* result, const uchar4* f0, const uchar4* f1,
  int width, int height, int pitch);

struct NoiseResult {
  uint64_t noise0, noise1;
  uint64_t noiseR0, noiseR1;
  uint64_t diff0, diff1;
};

struct UCFNoiseMeta {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x39EDF8,
  };
  int nMagicKey;
  int nVersion;

  int srcw, srch;
  int srcUVw, srcUVh;
  int noisew, noiseh;
  int noiseUVw, noiseUVh;

  UCFNoiseMeta()
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
  { }

  static const UCFNoiseMeta* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const UCFNoiseMeta* param = (const UCFNoiseMeta*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const UCFNoiseMeta* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

class KAnalyzeNoise : public KFMFilterBase
{
  PClip noiseclip;
  PClip superclip;

  UCFNoiseMeta meta;

  VideoInfo srcvi;
  VideoInfo padvi;

  void InitAnalyze(uint64_t* result, PNeoEnv env) {
    if (IS_CUDA) {
      kl_init_uint64 << <1, sizeof(NoiseResult) * 2 /sizeof(uint64_t) >> > (result);
      DEBUG_SYNC;
    }
    else {
      memset(result, 0x00, sizeof(NoiseResult) * 2);
    }
  }

  void AnalyzeNoise(uint64_t* resultY, uint64_t* resultUV, Frame noise0, Frame noise1, Frame noise2, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    const vpixel_t* src0Y = noise0.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src0U = noise0.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src0V = noise0.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* src1Y = noise1.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src1U = noise1.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src1V = noise1.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* src2Y = noise2.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src2U = noise2.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src2V = noise2.GetReadPtr<vpixel_t>(PLANAR_V);

    int pitchY = noise0.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = noise0.GetPitch<vpixel_t>(PLANAR_U);
    int width = noise0.GetWidth<vpixel_t>(PLANAR_Y);
    int widthUV = noise0.GetWidth<vpixel_t>(PLANAR_U);
    int height = noise0.GetHeight(PLANAR_Y);
    int heightUV = noise0.GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_analyze_noise << <blocks, threads >> >(resultY, src0Y, src1Y, src2Y, width, height, pitchY);
      DEBUG_SYNC;
      kl_analyze_noise << <blocksUV, threads >> >(resultUV, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_analyze_noise << <blocksUV, threads >> >(resultUV, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_analyze_noise(resultY, src0Y, src1Y, src2Y, width, height, pitchY);
      cpu_analyze_noise(resultUV, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      cpu_analyze_noise(resultUV, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
    }
  }

  void AnalyzeDiff(uint64_t* resultY, uint64_t* resultUV, Frame frame0, Frame frame1, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    const vpixel_t* src0Y = frame0.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src0U = frame0.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src0V = frame0.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* src1Y = frame1.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src1U = frame1.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src1V = frame1.GetReadPtr<vpixel_t>(PLANAR_V);

    int pitchY = frame0.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = frame0.GetPitch<vpixel_t>(PLANAR_U);
    int width = frame0.GetWidth<vpixel_t>(PLANAR_Y);
    int widthUV = frame0.GetWidth<vpixel_t>(PLANAR_U);
    int height = frame0.GetHeight(PLANAR_Y);
    int heightUV = frame0.GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_analyze_diff << <blocks, threads >> >(resultY, src0Y, src1Y, width, height, pitchY);
      DEBUG_SYNC;
      kl_analyze_diff << <blocksUV, threads >> >(resultUV, src0U, src1U, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_analyze_diff << <blocksUV, threads >> >(resultUV, src0V, src1V, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_analyze_diff(resultY, src0Y, src1Y, width, height, pitchY);
      cpu_analyze_diff(resultUV, src0U, src1U, widthUV, heightUV, pitchUV);
      cpu_analyze_diff(resultUV, src0V, src1V, widthUV, heightUV, pitchUV);
    }
  }

  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame noise0 = noiseclip->GetFrame(2 * n + 0, env);
    Frame noise1 = noiseclip->GetFrame(2 * n + 1, env);
    Frame noise2 = noiseclip->GetFrame(2 * n + 2, env);

    Frame f0padded;
    Frame f1padded;

    if (superclip) {
      f0padded = Frame(superclip->GetFrame(n, env), VPAD);
      f1padded = Frame(superclip->GetFrame(n + 1, env), VPAD);
    }
    else {
      Frame f0 = child->GetFrame(n, env);
      Frame f1 = child->GetFrame(n + 1, env);
      f0padded = Frame(env->NewVideoFrame(padvi), VPAD);
      f1padded = Frame(env->NewVideoFrame(padvi), VPAD);
      CopyFrame<uint8_t>(f0, f0padded, env);
      PadFrame<uint8_t>(f0padded, env);
      CopyFrame<uint8_t>(f1, f1padded, env);
      PadFrame<uint8_t>(f1padded, env);
    }

    Frame dst = env->NewVideoFrame(vi);

    NoiseResult* result = dst.GetWritePtr<NoiseResult>();

    InitAnalyze((uint64_t*)result, env);
    AnalyzeNoise(&result[0].noise0, &result[1].noise0, noise0, noise1, noise2, env);
    AnalyzeDiff(&result[0].diff0, &result[1].diff0, f0padded, f1padded, env);

    return dst.frame;
  }
public:
  KAnalyzeNoise(PClip src, PClip noise, PClip pad, IScriptEnvironment* env)
    : KFMFilterBase(src)
    , noiseclip(noise)
    , srcvi(vi)
    , padvi(vi)
    , superclip(pad)
  {
    if (srcvi.width & 3) env->ThrowError("[KAnalyzeNoise]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KAnalyzeNoise]: height must be multiple of 4");

    padvi.height += VPAD * 2;

    int out_bytes = sizeof(NoiseResult) * 2;
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);

    VideoInfo noisevi = noiseclip->GetVideoInfo();
    meta.srcw = srcvi.width;
    meta.srch = srcvi.height;
    meta.srcUVw = srcvi.width >> srcvi.GetPlaneWidthSubsampling(PLANAR_U);
    meta.srcUVh = srcvi.height >> srcvi.GetPlaneHeightSubsampling(PLANAR_U);
    meta.noisew = noisevi.width;
    meta.noiseh = noisevi.height;
    meta.noiseUVw = noisevi.width >> noisevi.GetPlaneWidthSubsampling(PLANAR_U);
    meta.noiseUVh = noisevi.height >> noisevi.GetPlaneHeightSubsampling(PLANAR_U);
    UCFNoiseMeta::SetParam(vi, &meta);

    if (!(GetDeviceTypes(src) & GetDeviceTypes(noise) & GetDeviceTypes(pad))) {
      env->ThrowError("[KAnalyzeNoise] Device unmatch. Three sources must be same device.");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT(n, env);
      //case 2:
      //  dst = InternalGetFrame<uint16_t>(n60, fmframe, frameType, env);
      //  break;
    default:
      env->ThrowError("[KAnalyzeNoise] Unsupported pixel format");
      break;
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KAnalyzeNoise(
      args[0].AsClip(),       // src
      args[1].AsClip(),       // noise
      args[2].Defined() ? args[2].AsClip() : nullptr,       // pad
      env
    );
  }
};

enum DECOMB_UCF_RESULT {
  DECOMB_UCF_CLEAN_1, // 1次判定で綺麗なフレームと判定
  DECOMB_UCF_CLEAN_2, // ノイズ判定で綺麗なフレームと判定
  DECOMB_UCF_USE_0,   // 1番目のフィールドを使うべき
  DECOMB_UCF_USE_1,   // 2番目のフィールドを使うべき
  DECOMB_UCF_NOISY,   // どっちも汚い
};

struct DecombUCFResult {
  DECOMB_UCF_RESULT flag;
  std::string message;
};

struct DecombUCFThreshScore {
  double y1, y2, y3, y4, y5;
  double x1, x2, x3, x4, x5;

  double calc(double x) const
  {
    return (x < x1) ? y1
      : (x < x2) ? ((y2 - y1)*x + x2*y1 - x1*y2) / (x2 - x1)
      : (x < x3) ? ((y3 - y2)*x + x3*y2 - x2*y3) / (x3 - x2)
      : (x < x4) ? ((y4 - y3)*x + x4*y3 - x3*y4) / (x4 - x3)
      : (x < x5) ? ((y5 - y4)*x + x5*y4 - x4*y5) / (x5 - x4)
      : y5;
  }
};

static DecombUCFThreshScore THRESH_SCORE_PARAM_TABLE[] = {
  {}, // 0（使われない）
  { 13,17,17,20,50,20,28,32,37,50 }, // 1
  { 14,18,20,40,50,19,28,36,42,50 },
  { 15,19,21,43,63,20,28,36,41,53 },
  { 15,20,23,43,63,20,28,36,41,53 },
  { 15,20,23,45,63,20,28,36,41,50 }, // 5(default)
  { 15,21,23,45,63,20,28,36,41,50 },
  { 15,22,24,45,63,20,28,35,41,50 },
  { 17,25,28,47,64,20,28,33,41,48 },
  { 20,32,38,52,66,21,30,36,40,48 },
  { 22,37,44,52,66,22,32,35,40,48 }, // 10
};

#define DecombUCF_PARAM_STR "[chroma]i[fd_thresh]f[th_mode]i[off_t]f[off_b]f" \
    "[namax_thresh]f[namax_diff]f[nrt1y]f[nrt2y]f[nrt2x]f[nrw]f[show]b" \
    "[y1]f[y2]f[y3]f[y4]f[y5]f[x1]f[x2]f[x3]f[x4]f[x5]f"

struct DecombUCFParam {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x40EDF8,
  };
  int nMagicKey;
  int nVersion;

  int chroma;       // [0-2] #(0:Y),(1:UV),(2:YUV) for noise detection
  double fd_thresh;    // [0-] #threshold of FieldDiff #fd_thresh = FieldDiff * 100 / (Width * Height)

                       // threshold
  int th_mode;      // [1-2:debug][3-7:normal][8-10:restricted] #preset of diff threshold. you can also specify threshold by x1-x5 y1-y5(need th_mode=0).
  double off_t;        // offset for diff threshold of top field (first field, top,diff<0)
  double off_b;        // offset for diff threshold of bottom field (second field, botom, 0<diff)

                       // reverse (chroma=0のみで機能。ノイズ量の絶対値が多過ぎる場合、映像効果と考えノイズの大きいフィールドを残す(小さいほうはブロックノイズによる平坦化))
  int namax_thresh; // 82 #MX:90 #[0-256] #disabled with chroma=1 #upper limit of max noise for Noise detaction (75-80-83)
  int namax_diff;   // 30-40 #disabled with chroma=1  #If average noise >= namax_thresh,  use namax_diff as diff threshold.

                    // NR
  double nrt1y;        // 28-29-30 #threshold for nr
  double nrt2y;        // 36-36.5-37 #exclusion range
  double nrt2x;        // 53-54-55 #exclusion range
  double nrw;          // 1-2 #diff weight for nr threshold

  bool show;

  DecombUCFThreshScore th_score;

  DecombUCFParam()
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
  { }

  static const DecombUCFParam* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const DecombUCFParam* param = (const DecombUCFParam*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const DecombUCFParam* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

static DecombUCFParam MakeParam(AVSValue args, int base, PNeoEnv env)
{
  DecombUCFParam param;
  param.chroma = args[base + 0].AsInt(1);
  param.fd_thresh = args[base + 1].AsFloat(128);
  param.th_mode = args[base + 2].AsInt(0);
  param.off_t = args[base + 3].AsFloat(0);
  param.off_b = args[base + 4].AsFloat(0);
  param.namax_thresh = args[base + 5].AsInt(82);
  param.namax_diff = args[base + 6].AsInt(38);
  param.nrt1y = args[base + 7].AsFloat(28);
  param.nrt2y = args[base + 8].AsFloat(36);
  param.nrt2x = args[base + 9].AsFloat(53.5);
  param.nrw = args[base + 10].AsFloat(2);
  param.show = args[base + 11].AsBool(false);

  // check param
  if (param.chroma < 0 || param.chroma > 2) {
    env->ThrowError("[DecombUCFParam]: chroma must be 0-2");
  }
  if (param.fd_thresh < 0) {
    env->ThrowError("[DecombUCFParam]: fd_thresh must be >=0");
  }
  if (param.th_mode < 0 || param.th_mode > 10) {
    env->ThrowError("[DecombUCFParam]: th_mode must be 0-10");
  }
  if (param.namax_thresh < 0 || param.namax_thresh > 256) {
    env->ThrowError("[DecombUCFParam]: namax_thresh should be in range 0-256");
  }

  base += 12;
  if (param.th_mode == 0) {
    DecombUCFThreshScore* def = &THRESH_SCORE_PARAM_TABLE[5];
    DecombUCFThreshScore th_score = {
      (float)args[base + 0].AsFloat((float)def->y1),
      (float)args[base + 1].AsFloat((float)def->y2),
      (float)args[base + 2].AsFloat((float)def->y3),
      (float)args[base + 3].AsFloat((float)def->y4),
      (float)args[base + 4].AsFloat((float)def->y5),
      (float)args[base + 5].AsFloat((float)def->x1),
      (float)args[base + 6].AsFloat((float)def->x2),
      (float)args[base + 7].AsFloat((float)def->x3),
      (float)args[base + 8].AsFloat((float)def->x4),
      (float)args[base + 9].AsFloat((float)def->x5)
    };
    param.th_score = th_score;
  }
  else {
    param.th_score = THRESH_SCORE_PARAM_TABLE[param.th_mode];
  }

  return param;
}

class KDecombUCFParam : public IClip
{
  VideoInfo vi;
  DecombUCFParam param;
public:
  KDecombUCFParam(DecombUCFParam param, IScriptEnvironment* env)
    : vi()
    , param(param)
  {
    DecombUCFParam::SetParam(vi, &this->param);
  }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) { return PVideoFrame(); }
  void __stdcall GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env) { }
  const VideoInfo& __stdcall GetVideoInfo() { return vi; }
  bool __stdcall GetParity(int n) { return true; }
  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return DEV_TYPE_CPU | DEV_TYPE_CUDA;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KDecombUCFParam(
      MakeParam(args, 0, env), // param
      env
    );
  }
};

DECOMB_UCF_RESULT CalcDecombUCF(
  const UCFNoiseMeta* meta,
  const DecombUCFParam* param,
  const NoiseResult* result0, // 1フレーム目
  const NoiseResult* result1, // 2フレーム目(second=falseならnullptr可)
  bool second,          // 
  std::string* message) // デバッグメッセージ
{
  double pixels = meta->srcw * meta->srch;
  //double pixelsUV = meta->srcUVw * meta->srcUVh;
  double noisepixels = meta->noisew * meta->noiseh;
  double noisepixelsUV = meta->noiseUVw * meta->noiseUVh * 2;

  // 1次判定フィールド差分
  double field_diff = (second 
    ? (result0[0].diff1 + result0[1].diff1)
    : (result0[0].diff0 + result0[1].diff0)) / (6 * pixels) * 100;

  // 絶対ノイズ量
  double noise_t_y = (second ? result0[0].noise1 : result0[0].noise0) / noisepixels;
  double noise_t_uv = (second ? result0[1].noise1 : result0[1].noise0) / noisepixelsUV;
  double noise_b_y = (second ? result1[0].noise0 : result0[0].noise1) / noisepixels;
  double noise_b_uv = (second ? result1[1].noise0 : result0[1].noise1) / noisepixelsUV;
  // 絶対ノイズ-平均(reverseで利用)
  double navg1_y = (noise_t_y + noise_b_y) / 2;
  double navg1_uv = (noise_t_uv + noise_b_uv) / 2;
  // 相対ノイズ-平均 [comp t,b](diff計算で利用)
  double navg2_y = (second ? result0[0].noiseR1 : result0[0].noiseR0) / noisepixels / 2;
  double navg2_uv = (second ? result0[1].noiseR1 : result0[1].noiseR0) / noisepixelsUV / 2;
  // 絶対ノイズ-符号付差分(diff計算で利用)
  double diff1_y = noise_t_y - noise_b_y;
  double diff1_uv = noise_t_uv - noise_b_uv;

  double diff1;     // 絶対ノイズ - 符号付差分
  double navg1;     // 絶対ノイズ平均(総ノイズ量判定用, 色差の細かい模様は滅多に見ない)
  double navg1_d;   // debug用
  double navg2;     // 相対ノイズ - 平均

  if (param->chroma == 0) {
    // Y
    diff1 = diff1_y;
    navg1_d = navg1 = navg1_y;
    navg2 = navg2_y;
  }
  else if (param->chroma == 1) {
    // UV
    diff1 = diff1_uv;
    navg1 = -1;
    navg1_d = navg1_uv;
    navg2 = navg2_uv;
  }
  else { // param->chroma == 2
    // YUV
    diff1 = (diff1_y + diff1_uv) / 2;
    navg1_d = navg1 = (navg1_y + navg1_uv) / 2;
    navg2 = (navg2_y + navg2_uv) / 2;
  }

  double absdiff1 = std::abs(diff1);
  double nmin1 = navg2 - absdiff1 / 2;
  double nmin = (nmin1 < 7) ? nmin1 * 4 : nmin1 + 21;
  double nmax = navg2 + absdiff1*param->nrw;
  double off_thresh = (diff1 < 0) ? param->off_t : param->off_b;
  double min_thresh = (navg1 < param->namax_thresh) 
    ? param->th_score.calc(nmin) + off_thresh 
    : param->namax_diff + off_thresh;
    // 符号付補正差分
  double diff = absdiff1 < 1.8 ? diff1 * 10
    : absdiff1 < 5 ? diff1 * 5 + (diff1 / absdiff1) * 9
    : absdiff1 < 10 ? diff1 * 2 + (diff1 / absdiff1) * 24
    : diff1 + (diff1 / absdiff1) * 34;

  DECOMB_UCF_RESULT result;
  if (std::abs(diff) < min_thresh) {
    result = ((nmax < param->nrt1y) || (param->nrt2x < navg1_d && nmax < param->nrt2y))
      ? DECOMB_UCF_CLEAN_2 : DECOMB_UCF_NOISY;
  }
  else if (navg1 < param->namax_thresh) {
    result = (diff < 0) ? DECOMB_UCF_USE_0 : DECOMB_UCF_USE_1;
  }
  else {
    result = (diff < 0) ? DECOMB_UCF_USE_1 : DECOMB_UCF_USE_0;
  }

  if (message) {
    char debug1_n_t[64];
    char debug1_n_b[64];
    if (param->chroma == 0) {
      sprintf_s(debug1_n_t, " [Y : %7f]", noise_t_y);
      sprintf_s(debug1_n_b, " [Y : %7f]", noise_b_y);
    }
    else if (param->chroma == 1) {
      sprintf_s(debug1_n_t, " [UV: %7f]", noise_t_uv);
      sprintf_s(debug1_n_b, " [UV: %7f]", noise_b_uv);
    }
    else {
      sprintf_s(debug1_n_t, " [Y : %7f] [UV: %7f]", noise_t_y, noise_t_uv);
      sprintf_s(debug1_n_b, " [Y : %7f] [UV: %7f]", noise_b_y, noise_b_uv);
    }
    char reschar = '-';
    char fdeq = '>';
    char noiseeq = '<';
    const char* field = "";
    if (field_diff < param->fd_thresh) {
      reschar = 'A';
      field = "notbob";
      fdeq = '<';
    }
    else if (result == DECOMB_UCF_CLEAN_2 || result == DECOMB_UCF_NOISY) {
      reschar = 'B';
      field = "notbob";
      if (result == DECOMB_UCF_NOISY) {
        noiseeq = '>';
      }
    }
    else {
      reschar = 'C';
      field = (result == DECOMB_UCF_USE_0) ? "First" : "Second";
    }
    const char* extra = "";
    if (result == DECOMB_UCF_NOISY) {
      extra = "NR";
    }
    else if (field_diff < param->fd_thresh && result != DECOMB_UCF_CLEAN_2) {
      extra = "NOT CLEAN ???";
    }
    else if (navg1 >= param->namax_thresh) {
      extra = "Reversed";
    }
    char buf[512];
    sprintf_s(buf,
      "[%c] %-6s  //  Fdiff =  %8f (FieldDiff %c %8f)\n"
      "                diff =  %8f  (NoiseDiff %c %.2f)\n"
      " Noise // First %s / Second %s\n"
      " navg1 : %.2f / nmin : %.2f / diff1 : %.3f / nrt : %.1f\n"
      "%s\n",
      reschar, field, field_diff, fdeq, param->fd_thresh,
      diff, noiseeq, min_thresh,
      debug1_n_t, debug1_n_b,
      navg1_d, nmin, diff1, nmax,
      extra);
    *message += buf;
  }

  return (field_diff < param->fd_thresh) ? DECOMB_UCF_CLEAN_1 : result;
}

// 24pクリップを分析して24pクリップに適用するフィルタ（オリジナルと同等）
class KDecombUCF : public KFMFilterBase
{
  PClip paramclip;
  PClip beforeclip;
  PClip afterclip;
  PClip noiseclip;
  PClip nrclip;

  DecombUCFInfo info;

  const UCFNoiseMeta* meta;
  const DecombUCFParam* param;
  PulldownPatterns patterns;

public:
  KDecombUCF(PClip clip24, PClip paramclip, PClip noiseclip, PClip beforeclip, PClip afterclip, PClip nrclip, IScriptEnvironment* env)
    : KFMFilterBase(clip24)
    , paramclip(paramclip)
    , noiseclip(noiseclip)
    , beforeclip(beforeclip)
    , afterclip(afterclip)
    , nrclip(nrclip)
    , info(30)
    , meta(UCFNoiseMeta::GetParam(noiseclip->GetVideoInfo(), env))
    , param(DecombUCFParam::GetParam(paramclip->GetVideoInfo(), env))
  {
    if (srcvi.width & 3) env->ThrowError("[KDecombUCF]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KDecombUCF]: height must be multiple of 4");

    // VideoInfoチェック
    VideoInfo vi24 = clip24->GetVideoInfo();
    VideoInfo vinoise = noiseclip->GetVideoInfo();
    VideoInfo vibefore = beforeclip->GetVideoInfo();
    VideoInfo viafter = afterclip->GetVideoInfo();

    if (vi24.num_frames != vinoise.num_frames)
      env->ThrowError("[KDecombUCF]: vi24.num_frames != vinoise.num_frames");
    if (vi24.num_frames * 2 != vibefore.num_frames)
      env->ThrowError("[KDecombUCF]: vi24.num_frames * 2 != vibefore.num_frames");
    if (vi24.num_frames * 2 != viafter.num_frames)
      env->ThrowError("[KDecombUCF]: vi24.num_frames * 2 != viafter.num_frames");
    if (vi24.width != vibefore.width)
      env->ThrowError("[KDecombUCF]: vi24.width != vibefore.width");
    if (vi24.width != viafter.width)
      env->ThrowError("[KDecombUCF]: vi24.width != viafterwidth");
    if (vi24.height != vibefore.height)
      env->ThrowError("[KDecombUCF]: vi24.height != vibefore.height");
    if (vi24.height != viafter.height)
      env->ThrowError("[KDecombUCF]: vi24.height != viafter.height");

    if (nrclip) {
      VideoInfo vinr = nrclip->GetVideoInfo();

      if (vi24.num_frames != vinr.num_frames)
        env->ThrowError("[KDecombUCF]: vi24.num_frames != vinr.num_frames");
      if (vi24.width != vinr.width)
        env->ThrowError("[KDecombUCF]: vi24.width != vinr.width");
      if (vi24.height != vinr.height)
        env->ThrowError("[KDecombUCF]: vi24.height != vinr.height");
    }

    auto devs = GetDeviceTypes(clip24);
    if (!(GetDeviceTypes(noiseclip) & devs)) {
      env->ThrowError("[KDecombUCF]: noiseclip device unmatch");
    }
    if (!(GetDeviceTypes(beforeclip) & devs)) {
      env->ThrowError("[KDecombUCF]: beforeclip device unmatch");
    }
    if (!(GetDeviceTypes(afterclip) & devs)) {
      env->ThrowError("[KDecombUCF]: afterclip device unmatch");
    }
    if (nrclip && !(GetDeviceTypes(nrclip) & devs)) {
      env->ThrowError("[KDecombUCF]: nrclip device unmatch");
    }
  }

  PVideoFrame __stdcall GetFrame(int n24, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PDevice cpu_device = env->GetDevice(DEV_TYPE_CPU, 0);

    Frame f0 = env->GetFrame(noiseclip, n24, cpu_device);
    const NoiseResult* result0 = f0.GetReadPtr<NoiseResult>();

    std::string message;
    auto result = CalcDecombUCF(meta, param,
      result0, nullptr, false, param->show ? &message : nullptr);

    if (param->show) {
      // messageを書いて返す
      Frame frame = child->GetFrame(n24, env);
      DrawText<uint8_t>(frame.frame, vi.BitsPerComponent(), 0, 0, message, env);
      return frame.frame;
    }

    if (result == DECOMB_UCF_USE_0) {
      return beforeclip->GetFrame(n24 * 2 + 0, env);
    }
    if (result == DECOMB_UCF_USE_1) {
      return afterclip->GetFrame(n24 * 2 + 1, env);
    }
    if (result == DECOMB_UCF_NOISY && nrclip) {
      return nrclip->GetFrame(n24, env);
    }
    return child->GetFrame(n24, env);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KDecombUCF(
      args[0].AsClip(),       // clip24
      args[1].AsClip(),       // paramclip
      args[2].AsClip(),       // noiseclip
      args[3].AsClip(),       // beforeclip
      args[4].AsClip(),       // afterclip
      args[5].Defined() ? args[5].AsClip() : nullptr,       // nrclip
      env
    );
  }
};

// 60iクリップを分析して24pクリップに適用するフィルタ
class KDecombUCF24 : public KFMFilterBase
{
  PClip paramclip;
  PClip fmclip;
  PClip beforeclip;
  PClip afterclip;
  PClip dweaveclip;
  PClip noiseclip;
  PClip nrclip;

  DecombUCFInfo info;

  const UCFNoiseMeta* meta;
  const DecombUCFParam* param;
  PulldownPatterns patterns;

public:
  KDecombUCF24(PClip clip24, PClip paramclip, PClip fmclip, PClip noiseclip, PClip beforeclip, PClip afterclip, PClip dweaveclip, PClip nrclip, IScriptEnvironment* env)
    : KFMFilterBase(clip24)
    , paramclip(paramclip)
    , fmclip(fmclip)
    , noiseclip(noiseclip)
    , beforeclip(beforeclip)
    , afterclip(afterclip)
    , dweaveclip(dweaveclip)
    , nrclip(nrclip)
    , info(24)
    , meta(UCFNoiseMeta::GetParam(noiseclip->GetVideoInfo(), env))
    , param(DecombUCFParam::GetParam(paramclip->GetVideoInfo(), env))
  {
    if (srcvi.width & 3) env->ThrowError("[KDecombUCF24]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KDecombUCF24]: height must be multiple of 4");

    // VideoInfoチェック
    VideoInfo vi24 = clip24->GetVideoInfo();
    VideoInfo vinoise = noiseclip->GetVideoInfo();
    VideoInfo vibefore = beforeclip->GetVideoInfo();
    VideoInfo viafter = afterclip->GetVideoInfo();
    VideoInfo vidw = dweaveclip->GetVideoInfo();
    VideoInfo vinr = nrclip ? nrclip->GetVideoInfo() : VideoInfo();

    // チェック
    CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);

    // fpsチェック
    vi24.MulDivFPS(5, 2);
    vinoise.MulDivFPS(2, 1);
    vibefore.MulDivFPS(1, 1);
    viafter.MulDivFPS(1, 1);
    vidw.MulDivFPS(1, 1);
    if (nrclip) {
      vinr.MulDivFPS(5, 2);
    }

    DecombUCFInfo::SetParam(vi, &info);

    if (vi24.fps_denominator != vinoise.fps_denominator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_denominator != vinoise.fps_denominator");
    if (vi24.fps_numerator != vinoise.fps_numerator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_numerator != vinoise.fps_numerator");
    if (vi24.fps_denominator != vibefore.fps_denominator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_denominator != vibefore.fps_denominator");
    if (vi24.fps_denominator != viafter.fps_denominator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_denominator != viafter.fps_denominator");
    if (vi24.fps_numerator != vibefore.fps_numerator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_numerator != vibefore.fps_numerator");
    if (vi24.fps_numerator != viafter.fps_numerator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_numerator != viafter.fps_numerator");
    if (vi24.fps_denominator != vidw.fps_denominator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_denominator != vidw.fps_denominator");
    if (vi24.fps_numerator != vidw.fps_numerator)
      env->ThrowError("[KDecombUCF24]: vi24.fps_numerator != vidw.fps_numerator");
    if (nrclip) {
      if (vi24.fps_denominator != vinr.fps_denominator)
        env->ThrowError("[KDecombUCF24]: vi24.fps_denominator != vinr.fps_denominator");
      if (vi24.fps_numerator != vinr.fps_numerator)
        env->ThrowError("[KDecombUCF24]: vi24.fps_numerator != vinr.fps_numerator");
    }

    // サイズチェック
    if (vi24.width != vibefore.width)
      env->ThrowError("[KDecombUCF24]: vi24.width != vibefore.width");
    if (vi24.width != viafter.width)
      env->ThrowError("[KDecombUCF24]: vi24.width != viafter.width");
    if (vi24.height != vibefore.height)
      env->ThrowError("[KDecombUCF24]: vi24.height != vibefore.height");
    if (vi24.height != viafter.height)
      env->ThrowError("[KDecombUCF24]: vi24.height != viafter.height");
    if (vi24.width != vidw.width)
      env->ThrowError("[KDecombUCF24]: vi24.width != vidw.width");
    if (vi24.height != vidw.height)
      env->ThrowError("[KDecombUCF24]: vi24.height != vidw.height");
    if (nrclip) {
      if (vi24.width != vinr.width)
        env->ThrowError("[KDecombUCF24]: vi24.width != vinr.width");
      if (vi24.height != vinr.height)
        env->ThrowError("[KDecombUCF24]: vi24.height != vinr.height");
    }

    if (!(GetDeviceTypes(fmclip) & DEV_TYPE_CPU)) {
      env->ThrowError("[KDecombUCF24]: fmclip must be CPU device");
    }
    if (!(GetDeviceTypes(noiseclip) & DEV_TYPE_CPU)) {
      env->ThrowError("[KDecombUCF24]: noiseclip must be CPU device");
    }

    auto devs = GetDeviceTypes(clip24);
    if (!(GetDeviceTypes(beforeclip) & devs)) {
      env->ThrowError("[KDecombUCF24]: beforeclip device unmatch");
    }
    if (!(GetDeviceTypes(afterclip) & devs)) {
      env->ThrowError("[KDecombUCF24]: afterclip device unmatch");
    }
    if (!(GetDeviceTypes(dweaveclip) & devs)) {
      env->ThrowError("[KDecombUCF24]: dweaveclip device unmatch");
    }
    if (nrclip && !(GetDeviceTypes(nrclip) & devs)) {
      env->ThrowError("[KDecombUCF24]: nrclip device unmatch");
    }
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    // CPU仮定のクリップがあるので
    if (cachehints == CACHE_GET_CHILD_DEV_TYPE) return DEV_TYPE_ANY;
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  PVideoFrame __stdcall GetFrame(int n24, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PDevice cpuDevice = env->GetDevice(DEV_TYPE_CPU, 0);

    int cycleIndex = n24 / 4;
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpuDevice)).GetReadPtr<KFMResult>());

    // 24pフレーム番号を取得
    Frame24Info frameInfo = patterns.GetFrame24(fm.pattern, n24);
    std::string message;

    bool cleanField[] = { true, true, true, true, true, true };
    for (int i = 0; i < frameInfo.numFields - 1; ++i) {
      int n60 = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex + i;
      Frame f0 = env->GetFrame(noiseclip, n60 / 2 + 0, cpuDevice);
      Frame f1 = env->GetFrame(noiseclip, n60 / 2 + 1, cpuDevice);
      const NoiseResult* result0 = f0.GetReadPtr<NoiseResult>();
      const NoiseResult* result1 = f1.GetReadPtr<NoiseResult>();

      std::string* mesptr = nullptr;
      if (param->show) {
        char buf[64];
        sprintf_s(buf, "24p Field: %d-%d(0-%d)\n", i, i + 1, frameInfo.numFields - 1);
        message += buf;
        mesptr = &message;
      }

      auto result = CalcDecombUCF(meta, param, result0, result1, (n60 & 1) != 0, mesptr);

      if (result == DECOMB_UCF_USE_0) {
        cleanField[i + 1] = false;
      }
      if (result == DECOMB_UCF_USE_1) {
        cleanField[i + 0] = false;
      }
      if (result == DECOMB_UCF_NOISY) {
        cleanField[i + 0] = false;
        cleanField[i + 1] = false;
      }
    }

    if (param->show) {
      // messageを書いて返す
      Frame frame = child->GetFrame(n24, env);
      DrawText<uint8_t>(frame.frame, vi.BitsPerComponent(), 0, 0, message, env);
      return frame.frame;
    }

    int n60start = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;

    if (std::find(cleanField, cleanField + frameInfo.numFields, false) != cleanField + frameInfo.numFields) {
      // 汚いフィールドが1枚以上ある
      for (int i = 0; i < frameInfo.numFields - 1; ++i) {
        if (cleanField[i + 0] && cleanField[i + 1]) {
          // 2枚連続できれいなフィールドがある -> きれいなフィールドで構成されたフレームを返す
          int n60 = n60start + i;
          return dweaveclip->GetFrame(n60, env);
        }
      }
      // 3フィールド以上あるのに2枚連続の綺麗なフィールドがない場合は、
      // そもそも全フィールドが汚い可能性が高いのでbobで更に汚くなるのを防ぐ
      if (frameInfo.numFields <= 2) {
        if (cleanField[0]) {
          // 1枚目のフィールドは綺麗 -> 後ろのフィールドは汚いので前のフィールドを使って補間
          return beforeclip->GetFrame(n60start, env);
        }
        else if (cleanField[1]) {
          // 2枚目のフィールドは綺麗 -> 前のフィールドは汚いので後ろのフィールドを使って補間
          return afterclip->GetFrame(n60start + 1, env);
        }
      }
      // きれいなフィールドがなかった -> NRを返す
      if (nrclip) {
        nrclip->GetFrame(n24, env);
      }
    }

    return child->GetFrame(n24, env);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KDecombUCF24(
      args[0].AsClip(),       // clip24
      args[1].AsClip(),       // paramclip
      args[2].AsClip(),       // fmclip
      args[3].AsClip(),       // noise
      args[4].AsClip(),       // before
      args[5].AsClip(),       // after
      args[6].AsClip(),       // dweave
      args[7].Defined() ? args[7].AsClip() : nullptr,       // nr
      env
    );
  }
};

// 60iクリップを分析して60pフラグを出力
class KDecombUCF60Flag : public GenericVideoFilter
{
  PClip paramclip;
  PClip showclip;
  float sc_thresh;
  float dup_thresh;

  const UCFNoiseMeta* meta;
  const DecombUCFParam* param;

  void GetFieldDiff(int nstart, double* diff, PNeoEnv env)
  {
    PDevice cpu_device = env->GetDevice(DEV_TYPE_CPU, 0);
    double pixels = meta->srcw * meta->srch;
    for (int i = 0; i < 4; ++i) {
      int n = nstart + i;
      Frame f0 = env->GetFrame(child, n / 2, cpu_device);
      const NoiseResult* result = f0.GetReadPtr<NoiseResult>();
      diff[i] = ((n & 1)
        ? (result[0].diff1 + result[1].diff1)
        : (result[0].diff0 + result[1].diff0)) / (6 * pixels) * 100;
    }
  }

public:
  KDecombUCF60Flag(PClip noiseclip, PClip paramclip, PClip showclip, float sc_thresh, float dup_thresh, IScriptEnvironment* env)
    : GenericVideoFilter(noiseclip)
    , paramclip(paramclip)
    , showclip(showclip)
    , sc_thresh(sc_thresh)
    , dup_thresh(dup_thresh)
    , meta(UCFNoiseMeta::GetParam(noiseclip->GetVideoInfo(), env))
    , param(DecombUCFParam::GetParam(paramclip->GetVideoInfo(), env))
  {
    if (!(GetDeviceTypes(noiseclip) & DEV_TYPE_CPU)) {
      env->ThrowError("[KDecombUCF60]: noiseclip must be CPU device");
    }
    if (param->show) {
      vi = showclip->GetVideoInfo();
    }
    else {
      // フレーム数、FPSを2倍
      vi.num_frames *= 2;
      vi.MulDivFPS(2, 1);
      // フレームは出力しないのでダミー
      vi.pixel_type = VideoInfo::CS_BGR32;
      vi.width = 4;
      vi.height = 1;
    }
  }

  PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PDevice cpu_device = env->GetDevice(DEV_TYPE_CPU, 0);

    DECOMB_UCF_RESULT replace_resluts[] = {
      DECOMB_UCF_USE_0,
      DECOMB_UCF_USE_1
    };

    int useFrame = n60;
    bool isDirty = false;
    std::string message;

    for (int i = 0; i < 2; ++i) {
      int n = n60 + i - 1;
      Frame f0 = env->GetFrame(child, n / 2 + 0, cpu_device);
      Frame f1 = env->GetFrame(child, n / 2 + 1, cpu_device);
      const NoiseResult* result0 = f0.GetReadPtr<NoiseResult>();
      const NoiseResult* result1 = f1.GetReadPtr<NoiseResult>();

      auto result = CalcDecombUCF(meta, param, result0, result1, (n & 1) != 0, nullptr);

      if (result == replace_resluts[i]) {
        double diff[4];
        if (i == 0) {
          // 前のフレームを使った方がいいと判定された
          GetFieldDiff(n60 - 3, diff, env);
          double sc = diff[3] / (std::max(diff[0], diff[1]) + 0.0001);
          if (sc > dup_thresh && diff[3] > sc_thresh) {
            // 前が静止
            useFrame = n60 - 1;
          }
          if (param->show) {
            char buf[200];
            const char* res = "";
            char eq = (sc > dup_thresh) ? '>' : '<';
            if (useFrame != n60) {
              res = "****** REPLACE FRAME WITH PREV ******";
            }
            sprintf_s(buf, "%s\nNEXT-SC: %7.2f %c %5.2f (%7.2f <= %7.2f, %7.2f)\n",
              res, sc, eq, dup_thresh, diff[3], diff[1], diff[0]);
            message += buf;
          }
        }
        else {
          // 後のフレームを使った方がいいと判定された
          GetFieldDiff(n60 - 1, diff, env);
          double sc = diff[0] / (std::max(diff[2], diff[3]) + 0.0001);
          if (sc > dup_thresh && diff[0] > sc_thresh) {
            // 後が静止
            useFrame = n60 + 1;
          }
          if (param->show) {
            char buf[200];
            const char* res = "";
            char eq = (sc > dup_thresh) ? '>' : '<';
            if (useFrame != n60) {
              res = "****** REPLACE FRAME WITH NEXT ******";
            }
            sprintf_s(buf, "%s\nPREV-SC: %7.2f %c %5.2f (%7.2f <= %7.2f, %7.2f)\n",
              res, sc, eq, dup_thresh, diff[0], diff[2], diff[3]);
            message += buf;
          }
        }
      }
      else if (result == DECOMB_UCF_NOISY) {
        isDirty = true;
      }
    }

    if (param->show) {
      // messageを書いて返す
      Frame frame = showclip->GetFrame(n60, env);
      if (useFrame == n60 && isDirty) {
        message = "******** !!! DIRTY FRAME !!! ********\n" + message;
      }
      else {
        message = "\n" + message;
      }
      DrawText<uint8_t>(frame.frame, vi.BitsPerComponent(), 0, 0, message, env);
      return frame.frame;
    }

    auto flag = DECOMB_UCF_NONE;
    if (useFrame == n60 && isDirty) {
      flag = DECOMB_UCF_NR;
    }
    else if (useFrame < n60) {
      flag = DECOMB_UCF_PREV;
    }
    else if (useFrame > n60) {
      flag = DECOMB_UCF_NEXT;
    }

    PVideoFrame res = env->NewVideoFrame(vi);
    res->SetProperty(DECOMB_UCF_FLAG_STR, (AVSMapValue)(int)flag);
    return res;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_CHILD_DEV_TYPE) {
      return DEV_TYPE_ANY;
    }
    else if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KDecombUCF60Flag(
      args[0].AsClip(),       // noiseclip
      args[1].AsClip(),       // paramclip
      args[2].AsClip(),       // showclip
      (float)args[3].AsFloat(256),       // sc_thresh
      (float)args[4].AsFloat(2.5),       // dup_factor
      env
    );
  }
};

// 60iクリップを分析して60pクリップに適用するフィルタ
class KDecombUCF60 : public KFMFilterBase
{
  PClip flagclip;
  PClip beforeclip;
  PClip afterclip;
  PClip nrclip;

  DecombUCFInfo info;

public:
  KDecombUCF60(PClip clip60, PClip flagclip, PClip beforeclip, PClip afterclip, PClip nrclip, IScriptEnvironment* env)
    : KFMFilterBase(clip60)
    , flagclip(flagclip)
    , beforeclip(beforeclip)
    , afterclip(afterclip)
    , nrclip(nrclip)
    , info(60)
  {
    if (srcvi.width & 3) env->ThrowError("[KDecombUCF60]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KDecombUCF60]: height must be multiple of 4");

    // VideoInfoチェック
    VideoInfo vi60 = clip60->GetVideoInfo();
    VideoInfo viflag = flagclip->GetVideoInfo();
    VideoInfo vibefore = beforeclip->GetVideoInfo();
    VideoInfo viafter = afterclip->GetVideoInfo();
    VideoInfo vinr = nrclip ? nrclip->GetVideoInfo() : VideoInfo();

    DecombUCFInfo::SetParam(vi, &info);

    // fpsチェック
    if (vi60.fps_denominator != viflag.fps_denominator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_denominator != viflag.fps_denominator");
    if (vi60.fps_numerator != viflag.fps_numerator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_numerator != viflag.fps_numerator");
    if (vi60.fps_denominator != vibefore.fps_denominator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_denominator != vibefore.fps_denominator");
    if (vi60.fps_denominator != viafter.fps_denominator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_denominator != viafter.fps_denominator");
    if (vi60.fps_numerator != vibefore.fps_numerator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_numerator != vibefore.fps_numerator");
    if (vi60.fps_numerator != viafter.fps_numerator)
      env->ThrowError("[KDecombUCF60]: vi60.fps_numerator != viafter.fps_numerator");
    if (nrclip) {
      if (vi60.fps_denominator != vinr.fps_denominator)
        env->ThrowError("[KDecombUCF60]: vi60.fps_denominator != vinr.fps_denominator");
      if (vi60.fps_numerator != vinr.fps_numerator)
        env->ThrowError("[KDecombUCF60]: vi60.fps_numerator != vinr.fps_numerator");
    }

    // サイズチェック
    if (vi60.width != vibefore.width)
      env->ThrowError("[KDecombUCF60]: vi60.num_frames != vibefore.num_frames");
    if (vi60.width != viafter.width)
      env->ThrowError("[KDecombUCF60]: vi60.num_frames != viafter.num_frames");
    if (vi60.height != vibefore.height)
      env->ThrowError("[KDecombUCF60]: vi60.num_frames != vibefore.num_frames");
    if (vi60.height != viafter.height)
      env->ThrowError("[KDecombUCF60]: vi60.num_frames != viafter.num_frames");
    if (nrclip) {
      if (vi60.width != vinr.width)
        env->ThrowError("[KDecombUCF60]: vi60.num_frames != viflag.num_frames");
      if (vi60.height != vinr.height)
        env->ThrowError("[KDecombUCF60]: vi60.num_frames != viflag.num_frames");
    }

    auto devs = GetDeviceTypes(clip60);
    if (!(GetDeviceTypes(beforeclip) & devs)) {
      env->ThrowError("[KDecombUCF60]: beforeclip device unmatch");
    }
    if (!(GetDeviceTypes(afterclip) & devs)) {
      env->ThrowError("[KDecombUCF60]: afterclip device unmatch");
    }
    if (nrclip && !(GetDeviceTypes(nrclip) & devs)) {
      env->ThrowError("[KDecombUCF60]: nrclip device unmatch");
    }
  }

  PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    DECOMB_UCF_FLAG centerFlag;
    DECOMB_UCF_FLAG sideFlag = DECOMB_UCF_NONE;
    PVideoFrame centerFrame;
    bool useNR = false;

    // 前後のフレームも考慮する
    for (int i = -1; i < 2; ++i) {
      PVideoFrame frame = flagclip->GetFrame(n60 + i, env);
      auto flag = (DECOMB_UCF_FLAG)frame->GetProperty(DECOMB_UCF_FLAG_STR, -1);
      if (flag == DECOMB_UCF_NR) {
        useNR = true;
      }
      else {
        if (flag == DECOMB_UCF_PREV || flag == DECOMB_UCF_NEXT) {
          if (i == -1) {
            sideFlag = DECOMB_UCF_NEXT;
          }
          else if (i == 1) {
            sideFlag = DECOMB_UCF_PREV;
          }
        }
      }
      if (i == 0) {
        centerFlag = flag;
        centerFrame = frame;
      }
    }

    // ディスパッチ
    PVideoFrame res;
    if (centerFlag == DECOMB_UCF_NR && nrclip) {
      res = nrclip->GetFrame(n60, env);
    }
    else if (centerFlag == DECOMB_UCF_PREV) {
      res = beforeclip->GetFrame(n60 - 1, env);
    }
    else if (centerFlag == DECOMB_UCF_NEXT) {
      res = afterclip->GetFrame(n60 + 1, env);
    }
    else if (useNR && nrclip) {
      res = nrclip->GetFrame(n60, env);
    }
    else if (sideFlag == DECOMB_UCF_PREV) {
      res = beforeclip->GetFrame(n60, env);
    }
    else if (sideFlag == DECOMB_UCF_NEXT) {
      res = afterclip->GetFrame(n60, env);
    }
    else {
      res = child->GetFrame(n60, env);
    }

    // フラグをコピーして返す
    env->CopyFrameProps(centerFrame, res);
    return res;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    PClip clip60 = args[0].AsClip();
    PClip paramclip = args[1].AsClip();
    PClip noiseclip = args[2].AsClip();
    PClip before = args[3].AsClip();
    PClip after = args[4].AsClip();
    AVSValue nrclip = args[5];
    AVSValue sc_thresh = args[6];
    AVSValue dup_thresh = args[7];

    std::vector<AVSValue> args2(5);
    args2[0] = noiseclip; // noiseclip
    args2[1] = paramclip; // paramclip
    args2[2] = clip60; // showclip
    args2[3] = sc_thresh; // sc_thresh
    args2[4] = dup_thresh; // dup_thresh
    PClip flagclip = env->Invoke("KDecombUCF60Flag", AVSValue(args2.data(), (int)args2.size())).AsClip();

    const DecombUCFParam* param = DecombUCFParam::GetParam(paramclip->GetVideoInfo(), env);
    if (param->show) {
      return flagclip;
    }
    return new KDecombUCF60(
      clip60, flagclip, before, after,
      nrclip.Defined() ? nrclip.AsClip() : nullptr,       // nrclip
      env
    );
  }
};

void AddFuncUCF(IScriptEnvironment* env)
{
  env->AddFunction("KCFieldDiff", "c[nt]f[chroma]b", KFieldDiff::CFunc, 0);
  env->AddFunction("KCFrameDiffDup", "c[chroma]b[blksize]i", KFrameDiffDup::CFunc, 0);

  env->AddFunction("KNoiseClip", "cc[nmin_y]i[range_y]i[nmin_uv]i[range_uv]i", KNoiseClip::Create, 0);
  env->AddFunction("KAnalyzeNoise", "cc[s4uper]c", KAnalyzeNoise::Create, 0);
  env->AddFunction("KDecombUCFParam", DecombUCF_PARAM_STR, KDecombUCFParam::Create, 0);
  env->AddFunction("KDecombUCF", "ccccc[nr]c", KDecombUCF::Create, 0);
  env->AddFunction("KDecombUCF24", "ccccccc[nr]c", KDecombUCF24::Create, 0);
  env->AddFunction("KDecombUCF60Flag", "cc[showclip]c[sc_thresh]f[dup_factor]f", KDecombUCF60Flag::Create, 0);
  env->AddFunction("KDecombUCF60", "ccccc[nr]c[sc_thresh]f[dup_factor]f", KDecombUCF60::Create, 0);
}
