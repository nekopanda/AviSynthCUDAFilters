
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>

#include "CommonFunctions.h"
#include "KFM.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"

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
  unsigned long long int CalcFieldDiff(PVideoFrame& frame, PVideoFrame& work, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(frame->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(frame->GetReadPtr(PLANAR_U));
    const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(frame->GetReadPtr(PLANAR_V));
    unsigned long long int* sum = reinterpret_cast<unsigned long long int*>(work->GetWritePtr());

    int pitchY = frame->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = frame->GetPitch(PLANAR_U) / sizeof(vpixel_t);
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
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame work = env->NewVideoFrame(workvi);

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
  int CalcFrameDiff(PVideoFrame& src0, PVideoFrame& src1, PVideoFrame& work, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* src0Y = reinterpret_cast<const vpixel_t*>(src0->GetReadPtr(PLANAR_Y));
    const vpixel_t* src0U = reinterpret_cast<const vpixel_t*>(src0->GetReadPtr(PLANAR_U));
    const vpixel_t* src0V = reinterpret_cast<const vpixel_t*>(src0->GetReadPtr(PLANAR_V));
    const vpixel_t* src1Y = reinterpret_cast<const vpixel_t*>(src1->GetReadPtr(PLANAR_Y));
    const vpixel_t* src1U = reinterpret_cast<const vpixel_t*>(src1->GetReadPtr(PLANAR_U));
    const vpixel_t* src1V = reinterpret_cast<const vpixel_t*>(src1->GetReadPtr(PLANAR_V));
    int* sumAbs = reinterpret_cast<int*>(work->GetWritePtr());
    int* sumSig = &sumAbs[block_pitch * blocks_h];
    int* maxSum = &sumSig[block_pitch * blocks_h];

    int pitchY = src0->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src0->GetPitch(PLANAR_U) / sizeof(vpixel_t);
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
    PVideoFrame src0 = child->GetFrame(clamp(n - 1, 0, vi.num_frames - 1), env);
    PVideoFrame src1 = child->GetFrame(clamp(n, 0, vi.num_frames - 1), env);
    PVideoFrame work = env->NewVideoFrame(workvi);

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

class KNoiseClip : public GenericVideoFilter
{
  PClip noiseclip;

  int range_y;
  int range_uv;
  int nmin_y;
  int nmin_uv;

  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame noise = noiseclip->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
    const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
    const vpixel_t* noiseY = reinterpret_cast<const vpixel_t*>(noise->GetReadPtr(PLANAR_Y));
    const vpixel_t* noiseU = reinterpret_cast<const vpixel_t*>(noise->GetReadPtr(PLANAR_U));
    const vpixel_t* noiseV = reinterpret_cast<const vpixel_t*>(noise->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width = src->GetRowSize(PLANAR_Y) / sizeof(vpixel_t);
    int widthUV = src->GetRowSize(PLANAR_U) / sizeof(vpixel_t);
    int height = src->GetHeight(PLANAR_Y);
    int heightUV = src->GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_noise_clip << <blocks, threads >> >(dstY, srcY, noiseY, width, height, pitchY, nmin_y, range_y);
      DEBUG_SYNC;
      kl_noise_clip << <blocksUV, threads >> >(dstU, srcU, noiseU, widthUV, height, pitchUV, nmin_uv, range_uv);
      DEBUG_SYNC;
      kl_noise_clip << <blocksUV, threads >> >(dstV, srcV, noiseV, widthUV, height, pitchUV, nmin_uv, range_uv);
      DEBUG_SYNC;
    }
    else {
      cpu_noise_clip(dstY, srcY, noiseY, width, height, pitchY, nmin_y, range_y);
      cpu_noise_clip(dstU, srcU, noiseU, widthUV, height, pitchUV, nmin_uv, range_uv);
      cpu_noise_clip(dstV, srcV, noiseV, widthUV, height, pitchUV, nmin_uv, range_uv);
    }

    return dst;
  }
public:
  KNoiseClip(PClip src, PClip noise,
    int nmin_y, int range_y, int nmin_uv, int range_uv, IScriptEnvironment* env)
    : GenericVideoFilter(src)
    , noiseclip(noise)
    , range_y(range_y)
    , range_uv(range_uv)
    , nmin_y(nmin_y)
    , nmin_uv(nmin_uv)
  {
    if (vi.width & 3) env->ThrowError("[KNoiseClip]: width must be multiple of 4");
    if (vi.height & 3) env->ThrowError("[KNoiseClip]: height must be multiple of 4");
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
      args[2].AsInt(1),       // fmclip
      args[3].AsInt(128),     // combeclip
      args[4].AsInt(1),       // thswitch
      args[5].AsInt(128),     // thpatch
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
  result[0] = sum0;
  result[1] = sum1;
  result[2] = sumR0;
  result[3] = sumR1;
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
  result[0] = sum0;
  result[1] = sum1;
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

class KAnalyzeNoise : public KFMFilterBase
{
  PClip noiseclip;

  VideoInfo srcvi;
  VideoInfo padvi;

  void InitAnalyze(uint64_t* result, PNeoEnv env) {
    if (IS_CUDA) {
      kl_init_uint64 << <1, 6 >> > (result);
      DEBUG_SYNC;
    }
    else {
      memset(result, 0x00, sizeof(NoiseResult));
    }
  }

  void AnalyzeNoise(uint64_t* result, PVideoFrame noise0, PVideoFrame noise1, PVideoFrame noise2, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    const vpixel_t* src0Y = reinterpret_cast<const vpixel_t*>(noise0->GetReadPtr(PLANAR_Y));
    const vpixel_t* src0U = reinterpret_cast<const vpixel_t*>(noise0->GetReadPtr(PLANAR_U));
    const vpixel_t* src0V = reinterpret_cast<const vpixel_t*>(noise0->GetReadPtr(PLANAR_V));
    const vpixel_t* src1Y = reinterpret_cast<const vpixel_t*>(noise1->GetReadPtr(PLANAR_Y));
    const vpixel_t* src1U = reinterpret_cast<const vpixel_t*>(noise1->GetReadPtr(PLANAR_U));
    const vpixel_t* src1V = reinterpret_cast<const vpixel_t*>(noise1->GetReadPtr(PLANAR_V));
    const vpixel_t* src2Y = reinterpret_cast<const vpixel_t*>(noise2->GetReadPtr(PLANAR_Y));
    const vpixel_t* src2U = reinterpret_cast<const vpixel_t*>(noise2->GetReadPtr(PLANAR_U));
    const vpixel_t* src2V = reinterpret_cast<const vpixel_t*>(noise2->GetReadPtr(PLANAR_V));

    int pitchY = noise0->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = noise0->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width = noise0->GetRowSize(PLANAR_Y) / sizeof(vpixel_t);
    int widthUV = noise0->GetRowSize(PLANAR_U) / sizeof(vpixel_t);
    int height = noise0->GetHeight(PLANAR_Y);
    int heightUV = noise0->GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_analyze_noise << <blocks, threads >> >(result, src0Y, src1Y, src2Y, width, height, pitchY);
      DEBUG_SYNC;
      kl_analyze_noise << <blocksUV, threads >> >(result, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_analyze_noise << <blocksUV, threads >> >(result, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_analyze_noise(result, src0Y, src1Y, src2Y, width, height, pitchY);
      cpu_analyze_noise(result, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      cpu_analyze_noise(result, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
    }
  }

  void AnalyzeDiff(uint64_t* result, PVideoFrame frame0, PVideoFrame frame1, PNeoEnv env)
  {
    typedef typename VectorType<uint8_t>::type vpixel_t;

    const vpixel_t* src0Y = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_Y));
    const vpixel_t* src0U = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_U));
    const vpixel_t* src0V = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_V));
    const vpixel_t* src1Y = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_Y));
    const vpixel_t* src1U = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_U));
    const vpixel_t* src1V = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_V));

    int pitchY = frame0->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = frame0->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width = frame0->GetRowSize(PLANAR_Y) / sizeof(vpixel_t);
    int widthUV = frame0->GetRowSize(PLANAR_U) / sizeof(vpixel_t);
    int height = frame0->GetHeight(PLANAR_Y);
    int heightUV = frame0->GetHeight(PLANAR_U);

    if (IS_CUDA) {
      dim3 threads(CALC_FIELD_DIFF_X, CALC_FIELD_DIFF_Y);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_analyze_diff << <blocks, threads >> >(result, src0Y, src1Y, width, height, pitchY);
      DEBUG_SYNC;
      kl_analyze_diff << <blocksUV, threads >> >(result, src0U, src1U, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_analyze_diff << <blocksUV, threads >> >(result, src0V, src1V, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_analyze_diff(result, src0Y, src1Y, width, height, pitchY);
      cpu_analyze_diff(result, src0U, src1U, widthUV, heightUV, pitchUV);
      cpu_analyze_diff(result, src0V, src1V, widthUV, heightUV, pitchUV);
    }
  }

  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    PVideoFrame f0 = child->GetFrame(n + 0, env);
    PVideoFrame f1 = child->GetFrame(n + 1, env);
    PVideoFrame noise0 = noiseclip->GetFrame(2 * n + 0, env);
    PVideoFrame noise1 = noiseclip->GetFrame(2 * n + 1, env);
    PVideoFrame noise2 = noiseclip->GetFrame(2 * n + 2, env);

    PVideoFrame f0padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame f1padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    uint64_t* result = reinterpret_cast<uint64_t*>(dst->GetWritePtr());

    // TODO: 切り出し
    CopyFrame<uint8_t>(f0, f0padded, env);
    PadFrame<uint8_t>(f0padded, env);
    CopyFrame<uint8_t>(f1, f1padded, env);
    PadFrame<uint8_t>(f1padded, env);

    InitAnalyze(result, env);
    AnalyzeNoise(result, noise0, noise1, noise2, env);
    AnalyzeDiff(result, f0padded, f1padded, env);

    return dst;
  }
public:
  KAnalyzeNoise(PClip src, PClip noise, IScriptEnvironment* env)
    : KFMFilterBase(src)
    , noiseclip(noise)
    , srcvi(vi)
    , padvi(vi)
  {
    if (srcvi.width & 3) env->ThrowError("[KAnalyzeNoise]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KAnalyzeNoise]: height must be multiple of 4");

    padvi.height += VPAD * 2;

    int out_bytes = sizeof(NoiseResult);
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);
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
      env
    );
  }
};

void AddFuncUCF(IScriptEnvironment* env)
{
  env->AddFunction("KCFieldDiff", "c[nt]f[chroma]b", KFieldDiff::CFunc, 0);
  env->AddFunction("KCFrameDiffDup", "c[chroma]b[blksize]i", KFrameDiffDup::CFunc, 0);
}
