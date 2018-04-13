
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
  CALC_FIELD_DIFF_X = 16,
  CALC_FIELD_DIFF_Y = 16,
  CALC_FIELD_DIFF_THREADS = CALC_FIELD_DIFF_X * CALC_FIELD_DIFF_Y
};

__global__ void kl_init_field_diff(unsigned long long int *sum)
{
  sum[threadIdx.x] = 0;
}

template <typename vpixel_t>
__global__ void kl_calculate_field_diff(const vpixel_t* ptr, int nt, int width, int height, int pitch, unsigned long long int *sum)
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
      kl_init_field_diff << <1, 1 >> > (sum);
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
    raw /= 6; // åvéZéÆÇ©ÇÁ

    int shift = vi.BitsPerComponent() - 8; // 8bitÇ…çáÇÌÇπÇÈ
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

    // dup232aÇæÇ∆Ç±Ç§ÇæÇØÇ«ÅAÇ±ÇÃåvéZéÆÇÕÇ®Ç©ÇµÇ¢Ç∆évÇ§ÇÃÇ≈èCê≥
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

void AddFuncUCF(IScriptEnvironment* env)
{
  env->AddFunction("KCFieldDiff", "c[nt]f[chroma]b", KFieldDiff::CFunc, 0);
  env->AddFunction("KCFrameDiffDup", "c[chroma]b[blksize]i", KFrameDiffDup::CFunc, 0);
}
