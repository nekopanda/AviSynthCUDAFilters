#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <stdio.h>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "nnedi3_kernel.h"

#include "CommonFunctions.h"
#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"

// commonのcppを取り入れる
#include "DeviceLocalData.cpp"

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
			CUDA_CHECK(cudaGetLastError()); \
      CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

template <typename T> struct VectorType {};

template <> struct VectorType<unsigned char> {
  typedef uchar4 type;
};

template <> struct VectorType<unsigned short> {
  typedef ushort4 type;
};

void OnCudaError(cudaError_t err) {
#if 1 // デバッグ用（本番は取り除く）
  printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
#endif
}

// width は Pad を含まない長さ
// block(2, -), threads(hPad, -)
template <typename pixel_t>
__global__ void kl_pad_h(pixel_t* ptr, int pitch, int hPad, int width, int height)
{
  bool isLeft = (blockIdx.x == 0);
  int x = threadIdx.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < height) {
    if (isLeft) {
      ptr[-(x + 1) + y * pitch] = ptr[(x + 1) + y * pitch];
    }
    else {
      ptr[(width + x) + y * pitch] = ptr[(width - (x + 2)) + y * pitch];
    }
  }
}

// height は Pad を含まない長さ
// block(-, 2), threads(-, vPad)
template <typename pixel_t>
__global__ void kl_pad_v(pixel_t* ptr, int pitch, int vPad, int width, int height)
{
  bool isTop = (blockIdx.y == 0);
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if (x < width) {
    if (isTop) {
      ptr[x - (y + 1) * pitch] = ptr[x + (y + 1) * pitch];
    }
    else {
      ptr[x + (height + y) * pitch] = ptr[x + (height - (y + 2)) * pitch];
    }
  }
}


template <typename vpixel_t>
__global__ void kl_copy(vpixel_t* dst, int dstpitch4, const vpixel_t* src, int srcpitch4, int width4, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width4 && y < height) {
    dst[x + y * dstpitch4] = src[x + y * srcpitch4];
  }
}


enum {
  PRE_BLOCK_W = 32,
  PRE_BLOCK_H = 16,
};

template <typename vpixel_t>
__global__ void kl_prescreening(
  vpixel_t* dst, int dstpitch4,
  const vpixel_t* __restrict__ ref, int refpitch4,
  const short4* __restrict__ ws, const float4* __restrict__ wf,
  uchar2* workNN, int* numblocks,
  int width4, int height, int val_min, int val_max)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = tx + ty * PRE_BLOCK_W;

  __shared__ short4 sws[64];
  __shared__ float4 swf[7];

  if (tid < 64) {
    sws[tid] = ws[tid];
  }
  if (tid < 7) {
    swf[tid] = wf[tid];
  }
  __syncthreads();

  int xbase = tx + blockIdx.x * PRE_BLOCK_W;
  int ybase = ty + blockIdx.y * PRE_BLOCK_H;

  float4 result = { 1,1,1,1 }; // 無効な値に初期化

  if (xbase < width4 && ybase < height) {

    int4 sum = { 0 };

    #pragma unroll
    for (int y = 0; y < 4; ++y) {
      #pragma unroll
      for (int x = 0; x < 5; ++x) {
        int4 v = to_int(ref[(x + xbase) + (y + ybase) * refpitch4]);
        if (x == 0) {
          sum += to_int(sws[0 + y * 16]) * v.z;
          sum += to_int(sws[1 + y * 16]) * v.w;
#if 0
          if (xbase == 0 && ybase == 270 && y == 3) {
            printf("src=(%d,%d,%d,%d)\n", v.x, v.y, v.z, v.w);
          }
#endif
        }
        else if (x < 4) {
          sum += to_int(sws[(x * 4 - 2) + y * 16]) * v.x;
          sum += to_int(sws[(x * 4 - 1) + y * 16]) * v.y;
          sum += to_int(sws[(x * 4 + 0) + y * 16]) * v.z;
          sum += to_int(sws[(x * 4 + 1) + y * 16]) * v.w;
        }
        else {
          sum += to_int(sws[14 + y * 16]) * v.x;
          sum += to_int(sws[15 + y * 16]) * v.y;
        }
      }
    }

    float4 t = to_float(sum) * swf[0] + swf[1];
    float4 val = t / (abs(t) + 1.0f);

    float4 sumf = { 0 };
    sumf += swf[2] * val.x;
    sumf += swf[3] * val.y;
    sumf += swf[4] * val.z;
    sumf += swf[5] * val.w;
    result = sumf + swf[6];
  }

  int num = 0;
  if (result.x <= 0.0f) ++num;
  if (result.y <= 0.0f) ++num;
  if (result.z <= 0.0f) ++num;
  if (result.w <= 0.0f) ++num;

#if 0
  if (xbase == 0 && ybase == 0) {
    printf("(0-3,0)=(%f,%f,%f,%f)\n", result.x, result.y, result.z, result.w);
  }
#endif

  int idx = num;
  __shared__ int sbuf[PRE_BLOCK_W * PRE_BLOCK_H / 32];
  dev_scan<int, PRE_BLOCK_W * PRE_BLOCK_H, AddReducer<int>>(tid, idx, sbuf);
  idx -= num;

  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  int workoff = bid * PRE_BLOCK_W * 4 * PRE_BLOCK_H;

  if (result.x <= 0.0f) workNN[workoff + idx++] = make_uchar2(tx * 4 + 0, ty);
  if (result.y <= 0.0f) workNN[workoff + idx++] = make_uchar2(tx * 4 + 1, ty);
  if (result.z <= 0.0f) workNN[workoff + idx++] = make_uchar2(tx * 4 + 2, ty);
  if (result.w <= 0.0f) workNN[workoff + idx++] = make_uchar2(tx * 4 + 3, ty);

  if (tid == PRE_BLOCK_W * PRE_BLOCK_H - 1) numblocks[bid] = idx;

  if (num < 4 && xbase < width4 && ybase < height) {
    int4 src3p = to_int(ref[(xbase + 2) + (ybase + 0) *refpitch4]);
    int4 src2 = to_int(ref[(xbase + 2) + (ybase + 1) *refpitch4]);
    int4 src4 = to_int(ref[(xbase + 2) + (ybase + 2) *refpitch4]);
    int4 src6 = to_int(ref[(xbase + 2) + (ybase + 3) *refpitch4]);

    // バイキュービック補間
    int4 tmp = clamp((((src2 + src4) * 19 - (src3p + src6) * 3 + 16) >> 5), val_min, val_max);

    // result <= 0のところはあとで計算するので必要ないが
    // 1個書き込んでも4個書き込んでも変わらないので（むしろ遅くなる可能性があるので）
    // 全部書き込む
    dst[xbase + ybase * dstpitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

enum {
  NN_BLOCK_W = 16,
  NN_BLOCK_H = 32,
};

template <typename pixel_t> struct ReadPixel8x6 {
  enum { K = 8 * 6 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    // txを2つに分割
    int yoff = (tx >> 3);
    int ttx = tx & 7;
    dst[ty][tx + 0] = src[ttx + (yoff + 0) * srcpitch];
    dst[ty][tx + 16] = src[ttx + (yoff + 2) * srcpitch];
    dst[ty][tx + 32] = src[ttx + (yoff + 4) * srcpitch];
  }
};

template <typename pixel_t> struct ReadPixel16x6 {
  enum { K = 16 * 6 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    for (int i = 0; i < 6; ++i) {
      dst[ty][tx + 16 * i] = src[tx + i * srcpitch];
    }
  }
};

template <typename pixel_t> struct ReadPixel32x6 {
  enum { K = 32 * 6 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    for (int i = 0; i < 6; ++i) {
      dst[ty][tx + 32 * i] = src[tx + i * srcpitch];
      dst[ty][tx + 16 + 32 * i] = src[tx + 16 + i * srcpitch];
    }
  }
};

template <typename pixel_t> struct ReadPixel48x6 {
  enum { K = 48 * 6 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    for (int i = 0; i < 6; ++i) {
      dst[ty][tx + 48 * i] = src[tx + i * srcpitch];
      dst[ty][tx + 16 + 48 * i] = src[tx + 16 + i * srcpitch];
      dst[ty][tx + 32 + 48 * i] = src[tx + 32 + i * srcpitch];
    }
  }
};

template <typename pixel_t> struct ReadPixel8x4 {
  enum { K = 8 * 4 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    // txを2つに分割
    int yoff = (tx >> 3);
    int ttx = tx & 7;
    dst[ty][tx + 0] = src[ttx + (yoff + 0) * srcpitch];
    dst[ty][tx + 16] = src[ttx + (yoff + 2) * srcpitch];
  }
};

template <typename pixel_t> struct ReadPixel16x4 {
  enum { K = 16 * 4 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    for (int i = 0; i < 4; ++i) {
      dst[ty][tx + 16 * i] = src[tx + i * srcpitch];
    }
  }
};

template <typename pixel_t> struct ReadPixel32x4 {
  enum { K = 32 * 4 };
  __device__ void operator()(int tx, int ty, pixel_t dst[][K], const pixel_t* src, int srcpitch) {
    for (int i = 0; i < 4; ++i) {
      dst[ty][tx + 32 * i] = src[tx + i * srcpitch];
      dst[ty][tx + 16 + 32 * i] = src[tx + 16 + i * srcpitch];
    }
  }
};

__device__ float dev_expf(float f)
{
  const float exp_lo = -80.0f;
  const float exp_hi = +80.0f;
  const float e0_mult = 12102203.161561486f;
  const float e0_bias = 1064866805.0f;
  union { int i; float f; } t;
  t.i = (int)(max(min(f, exp_hi), exp_lo)*e0_mult + e0_bias);
  return t.f;
}

template <typename pixel_t, int QUAL, int NN, typename READ>
__global__ void kl_compute_nn(pixel_t* dst, int dstpitch,
  const pixel_t* __restrict__ ref, int refpitch, const uchar2* __restrict__ workNN, const int* __restrict__ numblocks,
  const short2* __restrict__ weights, int wspitch, const float2* __restrict__ wf, int wfpitch, int val_min, int val_max)
{
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  int workoff = bid * PRE_BLOCK_W * 4 * PRE_BLOCK_H;

  int xbase = blockIdx.x * PRE_BLOCK_W * 4;
  int ybase = blockIdx.y * PRE_BLOCK_H;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  enum { K = READ::K };
  __shared__ pixel_t B[NN_BLOCK_H][K];
  __shared__ float avg[NN_BLOCK_H];    // mstd[0]
  __shared__ float var[NN_BLOCK_H];    // mstd[1]
  __shared__ float invvar[NN_BLOCK_H]; // mstd[2]

  int nb = numblocks[bid];
#if 0
  if (xbase == 0 && ybase == 0 && tx == 0 && ty == 0) {
    printf("nb=%d\n", nb);
  }
#endif
  for (int b = 0; b < nb; b += NN_BLOCK_H) {
    int x = xbase;
    int y = ybase;

    // read pixels
    if (b + ty < nb) {
      uchar2 xy = workNN[workoff + b + ty];
      x += xy.x;
      y += xy.y;
    }
#if 0
    if (workoff + b + ty == 0 && tx == 0 && ty == 0) {
      printf("(x,y)=(%d,%d)\n", x, y);
    }
    if (x == 0 && y == 0) {
      printf("HIT!!!\n");
    }
#endif
    READ readf;
    readf(tx, ty, B, &ref[x + y * refpitch], refpitch);
    __syncthreads();

    // sum pixels
    int sum = 0, sumsq = 0;
    for (int i = 0; i < K / NN_BLOCK_W; ++i) {
      pixel_t v = B[ty][tx + i * NN_BLOCK_W];
      sum += v; sumsq += v*v;
#if 0
      if (x == 0 && y == 0 && ty == 0) {
        printf("v[%d]=%d\n", tx + i * NN_BLOCK_W, v);
      }
#endif
    }
    dev_reduce_warp<int, NN_BLOCK_W, AddReducer<int>>(tx, sum);
    dev_reduce_warp<int, NN_BLOCK_W, AddReducer<int>>(tx, sumsq);
#if 0
    if (x == 0 && y == 0 && tx == 0 && ty == 0) {
      printf("sum,sumsq=%d,%d\n", sum, sumsq);
    }
#endif

    if (tx == 0) {
      const float scale = (float)(1.0 / (double)(K));
      float avg_ = sum * scale;
      float var_ = sumsq * scale - avg_ * avg_;
      float invvar_;

      if (var_ <= FLT_EPSILON) {
        var_ = 0;
        invvar_ = 0;
      }
      else {
        var_ = sqrtf(var_);
        invvar_ = 1.0f / var_;
      }

      avg[ty] = avg_;
      var[ty] = var_;
      invvar[ty] = invvar_;
    }
    __syncthreads();
#if 0
    if (x == 0 && y == 0 && tx == 0 && ty == 0) {
      printf("avg,var,invvar=%f,%f,%f\n", avg[ty], var[ty], invvar[ty]);
    }
#endif

    // compute network
    float result = 0.0f;
    for (int q = 0; q < QUAL; ++q) {
      float vsum = 0.0f, wsum = 0.0f;
      for (int i = 0; i < NN / NN_BLOCK_W; ++i) {
        int j = i * NN_BLOCK_W + tx;

        int2 sum = { 0 };
        for (int k = 0; k < K; ++k) {
          pixel_t v = B[ty][k];
          short2 w = weights[j + k * NN + q * wspitch];
          sum.x += v * w.x;
          sum.y += v * w.y;
        }

        float2 wf1 = wf[j + q * wfpitch];
        float2 wf2 = wf[j + NN + q * wfpitch];
        float res0 = (float)sum.x * wf1.x * invvar[ty] + wf2.x;
        float res1 = (float)sum.y * wf1.y * invvar[ty] + wf2.y;
#if 0
        if (x == 0 && y == 270 && tx == 0 && ty == 0) {
          printf("i=%d,%d,%d\n", i * NN_BLOCK_W + tx, sum.x, sum.y);
        }
#endif

        res0 = dev_expf(res0);

        vsum += res0 * (res1 / (1.0f + fabsf(res1)));
        wsum += res0;
#if 0
        if (x == 0 && y == 270 && tx == 0 && ty == 0) {
          printf("+++=%f,%f,%f\n", res0, res1, res0 * (res1 / (1.0f + fabsf(res1))));
        }
#endif
      }

      dev_reduce_warp<float, NN_BLOCK_W, AddReducer<float>>(tx, vsum);
      dev_reduce_warp<float, NN_BLOCK_W, AddReducer<float>>(tx, wsum);
#if 0
      if (x == 0 && y == 270 && tx == 0 && ty == 0) {
        printf("vsum,wsum=%f,%f\n", vsum, wsum);
      }
#endif

      if (tx == 0) {
        const float min_weight_sum = 1e-10f;
        if (wsum > min_weight_sum) result += ((5.0f*vsum) / wsum)*var[ty] + avg[ty];
        else result += avg[ty];
      }
    }

    // write result
    if (tx == 0 && b + ty < nb) {
      const float scale = (float)(1.0 / (double)QUAL);
      dst[x + y * dstpitch] = min(max((int)(result * scale + 0.5f), val_min), val_max);
    }

    __syncthreads(); // guard shared memory
  }
}

#define pixel_t uint8_t

void GetWorkBytes(int width, int height, int& nnBytes, int& blockBytes) {
  int blocks = nblocks(width, PRE_BLOCK_W * 4) * nblocks(height, PRE_BLOCK_H);
  nnBytes = blocks * PRE_BLOCK_W * 4 * PRE_BLOCK_H * sizeof(uchar2);
  blockBytes = blocks * sizeof(int);
}

void CopyPadCUDA(int pixelsize, pixel_t* ref, int refpitch, const pixel_t* src, int srcpitch, int width, int height, IScriptEnvironment2* env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
  {
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_copy<<<blocks, threads>>>((vpixel_t*)ref, refpitch / 4, (const vpixel_t*)src, srcpitch / 4, width / 4, height);
    DEBUG_SYNC;
  }
  {
    dim3 threads(32, 16);
    dim3 blocks(2, nblocks(height, threads.y));
    kl_pad_h<<<blocks, threads>>>(ref, refpitch, 32, width, height);
    DEBUG_SYNC;
  }
  {
    dim3 threads(32, 3);
    dim3 blocks(nblocks(width + 64, threads.x), 2);
    kl_pad_v<<<blocks, threads>>>(ref - 32, refpitch, 3, width + 64, height);
    DEBUG_SYNC;
  }
}

void BitBltCUDA(pixel_t* dst, int dstpitch, const pixel_t* src, int srcpitch, int width, int height, IScriptEnvironment2* env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_copy << <blocks, threads >> >((vpixel_t*)dst, dstpitch / 4, (const vpixel_t*)src, srcpitch / 4, width / 4, height);
  DEBUG_SYNC;
}

template <typename pixel_t>
class LaunchComputeNN
{
public:
  typedef void(*F)(
    dim3 preblock,
    pixel_t* dst, int dstpitch, const pixel_t* ref, int refpitch, uchar2* workNN, int* workBlock,
    const int16_t* weights1, int weights1pitch, int val_min, int val_max, IScriptEnvironment2* env);

  static F Get(int qual, int nns, int xdia, int ydia) {
    switch (qual) {
    case 1: return GetQ<1>(nns, xdia, ydia);
    case 2: return GetQ<2>(nns, xdia, ydia);
    default: return nullptr;
    }
  }

private:
  template <int QUAL, int NN, typename READ>
  static void Launch(
    dim3 preblock,
    pixel_t* dst, int dstpitch, const pixel_t* ref, int refpitch, uchar2* workNN, int* workBlock,
    const int16_t* weights1, int weights1pitch, int val_min, int val_max, IScriptEnvironment2* env)
  {
    const short2* ws = (const short2*)weights1;
    const float2* wf = (const float2 *)&ws[NN*READ::K];

    dim3 threads(NN_BLOCK_W, NN_BLOCK_H);

    kl_compute_nn<pixel_t, QUAL, NN, READ> << <preblock, threads >> >(
      dst, dstpitch, ref, refpitch, workNN, workBlock, ws, weights1pitch / 2, wf, weights1pitch / 4, val_min, val_max);
    DEBUG_SYNC;
  }

  template <int QUAL, int NN>
  static F GetQN(int xdia, int ydia) {
    if (ydia == 6) {
      switch (xdia) {
      case 8: return Launch<QUAL, NN, ReadPixel8x6<pixel_t>>;
      case 16: return Launch<QUAL, NN, ReadPixel16x6<pixel_t>>;
      case 32: return Launch<QUAL, NN, ReadPixel32x6<pixel_t>>;
      case 48: return Launch<QUAL, NN, ReadPixel48x6<pixel_t>>;
      default: return nullptr;
      }
    }
    else if (ydia == 4) {
      switch (xdia) {
      case 8: return Launch<QUAL, NN, ReadPixel8x4<pixel_t>>;
      case 16: return Launch<QUAL, NN, ReadPixel16x4<pixel_t>>;
      case 32: return Launch<QUAL, NN, ReadPixel32x4<pixel_t>>;
      default: return nullptr;
      }
    }
    return nullptr;
  }

  template <int QUAL>
  static F GetQ(int nns, int xdia, int ydia) {
    switch (nns) {
    case 16: return GetQN<QUAL, 16>(xdia, ydia);
    case 32: return GetQN<QUAL, 32>(xdia, ydia);
    case 64: return GetQN<QUAL, 64>(xdia, ydia);
    case 128: return GetQN<QUAL, 128>(xdia, ydia);
    case 256: return GetQN<QUAL, 256>(xdia, ydia);
    default: return nullptr;
    }
  }
};


void EvalCUDA(int pixelsize, int bits_per_pixel,
  pixel_t* dst, int dstpitch, const pixel_t* ref, int refpitch, int width, int height,
  const int16_t* weights0, const int16_t* weights1, int weights1pitch, 
  uint8_t* workNN_, uint8_t* workBlock_,
  int range_mode, int qual, int nns, int xdia, int ydia, IScriptEnvironment2* env)
{
  typedef typename VectorType<pixel_t>::type vpixel_t;

  int bitsm8 = bits_per_pixel - 8;
  pixel_t val_min, val_max;

  switch (range_mode)
  {
  case 1:
    val_min = 0; val_max = (1 << bits_per_pixel) - 1;
    break;
  case 2:
    val_min = (16 << bitsm8); val_max = (235 << bitsm8);
    break;
  case 3:
    val_min = (16 << bitsm8); val_max = (240 << bitsm8);
    break;
  case 4:
    val_min = (16 << bitsm8); val_max = (1 << bits_per_pixel) - 1;
    break;
  default:
    val_min = 0; val_max = (1 << bits_per_pixel) - 1;
    break;
  }

  int dstpitch4 = dstpitch / 4;
  int refpitch4 = refpitch / 4;
  int width4 = width / 4;

  uchar2* workNN = (uchar2*)workNN_;
  int* workBlock = (int*)workBlock_;

  const short4* ws = (const short4*)weights0;
  const float4* wf = (float4*)&ws[64];
    
  dim3 threads(PRE_BLOCK_W, PRE_BLOCK_H);
  dim3 blocks(nblocks(width4, PRE_BLOCK_W), nblocks(height, PRE_BLOCK_H));
  kl_prescreening<vpixel_t><<<blocks, threads>>>(
    (vpixel_t*)dst, dstpitch4, (const vpixel_t*)(ref - refpitch - 8), refpitch4,
    ws, wf, workNN, workBlock, width4, height, val_min, val_max);
  DEBUG_SYNC;

  typename LaunchComputeNN<pixel_t>::F launch_compute = LaunchComputeNN<pixel_t>::Get(qual, nns, xdia, ydia);

  if (launch_compute == nullptr) {
    env->ThrowError("[KNNEDI3] インスタンス化してないパラメータです");
  }

  const pixel_t *refpp = ref - (((ydia >> 1) - 1)*refpitch + ((xdia >> 1) - 1));
  launch_compute(blocks, dst, dstpitch, refpp, refpitch,
    workNN, workBlock, weights1, weights1pitch, val_min, val_max, env);
}
