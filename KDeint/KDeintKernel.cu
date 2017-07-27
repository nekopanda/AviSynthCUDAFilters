#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "KDeintKernel.h"

#include "ReduceKernel.cuh"

/////////////////////////////////////////////////////////////////////////////
// COPY
/////////////////////////////////////////////////////////////////////////////

template <typename pixel_t>
__global__ void kl_copy(
  pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * dst_pitch] = src[x + y * src_pitch];
  }
}

template <typename pixel_t>
void KDeintKernel::Copy(
  pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_copy<pixel_t> << <blocks, threads, 0, stream >> > (
    dst, dst_pitch, src, src_pitch, width, height);
  DebugSync();
}

template void KDeintKernel::Copy<uint8_t>(
  uint8_t* dst, int dst_pitch, const uint8_t* src, int src_pitch, int width, int height);
template void KDeintKernel::Copy<uint16_t>(
  uint16_t* dst, int dst_pitch, const uint16_t* src, int src_pitch, int width, int height);
template void KDeintKernel::Copy<int16_t>(
  int16_t* dst, int dst_pitch, const int16_t* src, int src_pitch, int width, int height);
template void KDeintKernel::Copy<int32_t>(
  int32_t* dst, int dst_pitch, const int32_t* src, int src_pitch, int width, int height);


/////////////////////////////////////////////////////////////////////////////
// PadFrame
/////////////////////////////////////////////////////////////////////////////

// width は Pad を含まない長さ
// block(2, -), threads(hPad, -)
template <typename pixel_t>
__global__ void kl_pad_frame_h(pixel_t* ptr, int pitch, int hPad, int width, int height)
{
  bool isLeft = (blockIdx.x == 0);
  int x = threadIdx.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < height) {
    if (isLeft) {
      ptr[x + y * pitch] = ptr[hPad + y * pitch];
    }
    else {
      ptr[(hPad + width + x) + y * pitch] = ptr[(hPad + width) + y * pitch];
    }
  }
}

// height は Pad を含まない長さ
// block(-, 2), threads(-, vPad)
template <typename pixel_t>
__global__ void kl_pad_frame_v(pixel_t* ptr, int pitch, int vPad, int width, int height)
{
  bool isTop = (blockIdx.y == 0);
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if (x < width) {
    if (isTop) {
      ptr[x + y * pitch] = ptr[x + vPad * pitch];
    }
    else {
      ptr[x + (vPad + height + y) * pitch] = ptr[x + (vPad + height) * pitch];
    }
  }
}

template<typename pixel_t>
void KDeintKernel::PadFrame(pixel_t *ptr, int pitch, int hPad, int vPad, int width, int height)
{
  { // H方向
    dim3 threads(hPad, 32);
    dim3 blocks(2, nblocks(height, threads.y));
    kl_pad_frame_h<pixel_t> << <blocks, threads, 0, stream >> > (
      ptr + vPad * pitch, pitch, hPad, width, height);
    DebugSync();
  }
  { // V方向（すでにPadされたH方向分も含む）
    dim3 threads(32, vPad);
    dim3 blocks(nblocks(width + hPad * 2, threads.x), 2);
    kl_pad_frame_v<pixel_t> << <blocks, threads, 0, stream >> > (
      ptr, pitch, vPad, width + hPad * 2, height);
    DebugSync();
  }
}

template void KDeintKernel::PadFrame<uint8_t>(
  uint8_t *ptr, int pitch, int hPad, int vPad, int width, int height);
template void KDeintKernel::PadFrame<uint16_t>(
  uint16_t *ptr, int pitch, int hPad, int vPad, int width, int height);


/////////////////////////////////////////////////////////////////////////////
// Wiener
/////////////////////////////////////////////////////////////////////////////

// so called Wiener interpolation. (sharp, similar to Lanczos ?)
// invarint simplified, 6 taps. Weights: (1, -5, 20, 20, -5, 1)/32 - added by Fizick
template<typename pixel_t>
__global__ void kl_vertical_wiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nWidth) {
    if (y < 2) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[x + (y + 1) * nSrcPitch] + 1) >> 1;
    }
    else if (y < nHeight - 4) {
      pDst[x + y * nDstPitch] = min(max_pixel_value, max(0,
        (pSrc[x + (y - 2) * nSrcPitch]
          + (-(pSrc[x + (y - 1) * nSrcPitch]) + (pSrc[x + y * nSrcPitch] << 2) +
          (pSrc[x + (y + 1) * nSrcPitch] << 2) - (pSrc[x + (y + 2) * nSrcPitch])) * 5
          + (pSrc[x + (y + 3) * nSrcPitch]) + 16) >> 5));
    }
    else if (y < nHeight - 1) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[x + (y + 1) * nSrcPitch] + 1) >> 1;
    }
    else if (y < nHeight) {
      // last row
      pDst[x + y * nDstPitch] = pSrc[x + y * nSrcPitch];
    }
  }
}

template<typename pixel_t>
void KDeintKernel::VerticalWiener(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
{
  const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

  dim3 threads(32, 16);
  dim3 blocks(nblocks(nWidth, threads.x), nblocks(nHeight, threads.y));
  kl_vertical_wiener<pixel_t> << <blocks, threads, 0, stream >> > (
    pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight, max_pixel_value);
  DebugSync();
}

template<typename pixel_t>
__global__ void kl_horizontal_wiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < nHeight) {
    if (x < 2) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[(x + 1) + y * nSrcPitch] + 1) >> 1;
    }
    else if (x < nWidth - 4) {
      pDst[x + y * nDstPitch] = min(max_pixel_value, max(0,
        (pSrc[(x - 2) + y * nSrcPitch]
          + (-(pSrc[(x - 1) + y * nSrcPitch]) + (pSrc[x + y * nSrcPitch] << 2) +
          (pSrc[(x + 1) + y * nSrcPitch] << 2) - (pSrc[(x + 2) + y * nSrcPitch])) * 5
          + (pSrc[(x + 3) + y * nSrcPitch]) + 16) >> 5));
    }
    else if (x < nWidth - 1) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[(x + 1) + y * nSrcPitch] + 1) >> 1;
    }
    else if (x < nWidth) {
      // last column
      pDst[x + y * nDstPitch] = pSrc[x + y * nSrcPitch];
    }
  }
}

template<typename pixel_t>
void KDeintKernel::HorizontalWiener(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
{
  const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

  dim3 threads(32, 16);
  dim3 blocks(nblocks(nWidth, threads.x), nblocks(nHeight, threads.y));
  kl_horizontal_wiener<pixel_t> << <blocks, threads, 0, stream >> > (
    pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight, max_pixel_value);
  DebugSync();
}


template void KDeintKernel::VerticalWiener<uint8_t>(
  uint8_t *pDst, const uint8_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);
template void KDeintKernel::VerticalWiener<uint16_t>(
  uint16_t *pDst, const uint16_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);
template void KDeintKernel::HorizontalWiener<uint8_t>(
  uint8_t *pDst, const uint8_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);
template void KDeintKernel::HorizontalWiener<uint16_t>(
  uint16_t *pDst, const uint16_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel);


/////////////////////////////////////////////////////////////////////////////
// RB2BilinearFilter
/////////////////////////////////////////////////////////////////////////////

enum {
  RB2B_BILINEAR_W = 32,
  RB2B_BILINEAR_H = 16,
};

// BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing - Fizick
// threads=(RB2B_BILINEAR_W,RB2B_BILINEAR_H)
// nblocks=(nblocks(nWidth*2, RB2B_BILINEAR_W - 2),nblocks(nHeight,RB2B_BILINEAR_H))
template<typename pixel_t>
__global__ void kl_RB2B_bilinear_filtered(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight)
{
  __shared__ pixel_t tmp[RB2B_BILINEAR_H][RB2B_BILINEAR_W];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Verticalを実行
  // Horizontalで参照するため両端1列ずつ余分に実行
  int x = tx - 1 + blockIdx.x * (RB2B_BILINEAR_W - 2);
  int y = ty + blockIdx.y * RB2B_BILINEAR_H;
  int y2 = y * 2;

  if (x >= 0 && x < nWidth * 2) {
    if (y < 1) {
      tmp[ty][tx] = (pSrc[x + y2 * nSrcPitch] + pSrc[x + (y2 + 1) * nSrcPitch] + 1) / 2;
    }
    else if (y < nHeight - 1) {
      tmp[ty][tx] = (pSrc[x + (y2 - 1) * nSrcPitch]
        + pSrc[x + y2 * nSrcPitch] * 3
        + pSrc[x + (y2 + 1) * nSrcPitch] * 3
        + pSrc[x + (y2 + 2) * nSrcPitch] + 4) / 8;
    }
    else if (y < nHeight) {
      tmp[ty][tx] = (pSrc[x + y2 * nSrcPitch] + pSrc[x + (y2 + 1) * nSrcPitch] + 1) / 2;
    }
  }

  __syncthreads();

  // Horizontalを実行
  x = tx + blockIdx.x * ((RB2B_BILINEAR_W - 2) / 2);
  int tx2 = tx * 2;

  if (tx < ((RB2B_BILINEAR_W - 2) / 2) && y < nHeight) {
    // tmpは[0][1]が原点であることに注意
    if (x < 1) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2 + 1] + tmp[ty][tx2 + 2] + 1) / 2;
    }
    else if (x < nWidth - 1) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2]
        + tmp[ty][tx2 + 1] * 3
        + tmp[ty][tx2 + 2] * 3
        + tmp[ty][tx2 + 3] + 4) / 8;
    }
    else if (x < nWidth) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2 + 1] + tmp[ty][tx2 + 2] + 1) / 2;
    }
  }
}

template<typename pixel_t>
void KDeintKernel::RB2BilinearFiltered(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight)
{
  dim3 threads(RB2B_BILINEAR_W, RB2B_BILINEAR_H);
  dim3 blocks(nblocks(nWidth*2, RB2B_BILINEAR_W - 2), nblocks(nHeight, RB2B_BILINEAR_H));
  kl_RB2B_bilinear_filtered<pixel_t> << <blocks, threads, 0, stream >> > (
    pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight);
  DebugSync();
}

template void KDeintKernel::RB2BilinearFiltered<uint8_t>(
  uint8_t *pDst, const uint8_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight);
template void KDeintKernel::RB2BilinearFiltered<uint16_t>(
  uint16_t *pDst, const uint16_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight);



/////////////////////////////////////////////////////////////////////////////
// SearchMV
/////////////////////////////////////////////////////////////////////////////


typedef int sad_t; // 後でfloatにする

enum {
  SRCH_DIMX = 128
};

struct SearchBlock {
  // [0-3]: nDxMax, nDyMax, nDxMin, nDyMin （MaxはMax-1にしておく）
  // [4-9]: Left predictor, Up predictor, bottom-right predictor(from coarse level)
  // 無効なところは作らないようにする（最低でもどれか１つは有効なので無効なろところはそのインデックスで埋める）
  // [10-11]: predictor の x, y
  int data[12];
  // [0-3]: penaltyZero, penaltyGlobal, 1(penaltyPredictor), penaltyNew
  // [4]: lambda
  sad_t dataf[5];
};

#define CLIP_RECT data
#define REF_VECTOR_INDEX (&data[4])
#define PRED_X data[10]
#define PRED_Y data[11]
#define PENALTIES dataf
#define PENALTY_NEW dataf[3]
#define LAMBDA dataf[4]

#define LARGE_COST INT_MAX

struct CostResult {
  sad_t cost;
  short2 xy;
};

__device__ void dev_clip_mv(short2& v, const int* rect)
{
  v.x = (v.x > rect[0]) ? rect[0] : (v.x < rect[2]) ? rect[2] : v.x;
  v.y = (v.y > rect[1]) ? rect[1] : (v.y < rect[3]) ? rect[3] : v.y;
}

__device__ bool dev_check_mv(int x, int y, const int* rect)
{
  return (x <= rect[0]) & (y <= rect[1]) & (x >= rect[2]) & (y >= rect[3]);
}

__device__ int dev_max(int a, int b, int c) {
  int ab = (a > b) ? a : b;
  return (ab > c) ? ab : c;
}

__device__ int dev_min(int a, int b, int c) {
  int ab = (a < b) ? a : b;
  return (ab < c) ? ab : c;
}

__device__ int dev_sq_norm(int ax, int ay, int bx, int by) {
  return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
}

// pRef は ブロックオフセット分を予め移動させておいたポインタ
// vx,vy は サブピクセルも含めたベクトル
template <typename pixel_t, int NPEL>
__device__ const pixel_t* dev_get_ref_block(const pixel_t* pRef, int nPitch, int nImgPitch, int vx, int vy)
{
  if (NPEL != 1) {
    int sx = vx % NPEL;
    int sy = vy % NPEL;
    int si = sx + sy * NPEL;
    int x = vx / NPEL;
    int y = vy / NPEL;
    return &pRef[x + y * nPitch + si * nImgPitch];
  }
  else {
    return &pRef[vx + vy * nPitch];
  }
}

template <typename pixel_t, int BLK_SIZE>
__device__ sad_t dev_calc_sad(
  int wi,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* pRefY, const pixel_t* pRefU, const pixel_t* pRefV,
  int nPitchY, int nPitchU, int nPitchV)
{
  int sad = 0;
  if (BLK_SIZE == 16) {
    // ブロックサイズがスレッド数と一致
    int yx = wi;
    for (int yy = 0; yy < BLK_SIZE; ++yy) { // 16回ループ
      sad = __sad(pSrcY[yx + yy * BLK_SIZE], pRefY[yx + yy * nPitchY], sad);
    }
    // UVは8x8
    int uvx = wi % 8;
    int uvy = wi / 8;
    for (int t = 0; t < 4; ++t, uvy += 2) { // 4回ループ
      sad = __sad(pSrcU[uvx + uvy * BLK_SIZE], pRefU[uvx + uvy * nPitchU], sad);
      sad = __sad(pSrcV[uvx + uvy * BLK_SIZE], pRefV[uvx + uvy * nPitchV], sad);
    }
  }
  else if (BLK_SIZE == 32) {
    // 32x32
    int yx = wi;
    for (int yy = 0; yy < BLK_SIZE; ++yy) { // 32回ループ
      sad = __sad(pSrcY[yx + yy * BLK_SIZE], pRefY[yx + yy * nPitchY], sad);
      sad = __sad(pSrcY[yx + 16 + yy * BLK_SIZE], pRefY[yx + 16 + yy * nPitchY], sad);
    }
    // ブロックサイズがスレッド数と一致
    int uvx = wi;
    for (int uvy = 0; uvy < BLK_SIZE; ++uvy) { // 16回ループ
      sad = __sad(pSrcU[uvx + uvy * BLK_SIZE], pRefU[uvx + uvy * nPitchU], sad);
      sad = __sad(pSrcV[uvx + uvy * BLK_SIZE], pRefV[uvx + uvy * nPitchV], sad);
    }
  }
  return dev_reduce_warp<int, 16, AddReducer>(sad, wi);
}

// MAX - (MAX/4) <= (結果の個数) <= MAX であること
// スレッド数は (結果の個数) - MAX/2
template <int MAX>
__device__ void dev_reduce_result(CostResult* tmp_, int tid)
{
  volatile CostResult* tmp = (volatile CostResult*)tmp_;
  if(MAX >= 16) tmp[tid] = (tmp[tid].cost < tmp[tid + 8].cost) ? tmp[tid] : tmp[tid + 8];
  tmp[tid] = (tmp[tid].cost < tmp[tid + 4].cost) ? tmp[tid] : tmp[tid + 4];
  tmp[tid] = (tmp[tid].cost < tmp[tid + 2].cost) ? tmp[tid] : tmp[tid + 2];
  tmp[tid] = (tmp[tid].cost < tmp[tid + 1].cost) ? tmp[tid] : tmp[tid + 1];
}

// __syncthreads()を呼び出しているので全員で呼ぶ
template <typename pixel_t, int BLK_SIZE, int NPEL>
__device__ void dev_expanding_search_1(
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  int2 area[] = {
    { -1, -1 },
    { 0, -1 },
    { 1, -1 },
    { -1, 0 },
    { 1, 0 },
    { -1, 1 },
    { 0, 1 },
    { 1, 1 }
  };

  __shared__ bool isVectorOK[8];
  __shared__ CostResult result[8];
  __shared__ const pixel_t* pRefY[8];
  __shared__ const pixel_t* pRefU[8];
  __shared__ const pixel_t* pRefV[8];

  if (tx < 8) {
    int x = result[tx].xy.x = cx + area[tx].x;
    int y = result[tx].xy.y = cy + area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }

    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x / 2, y / 2);
    pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x / 2, y / 2);
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += (sad * PENALTY_NEW) >> 8;
    }
  }

  __syncthreads();

  // 結果集約
  if (tx < 4) { // reduceは8-4=4スレッドで呼ぶ
    dev_reduce_result<8>(result, tx);

    if (tx == 0) { // tx == 0は最後のデータを書き込んでいるのでアクセスOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

// __syncthreads()を呼び出しているので全員で呼ぶ
template <typename pixel_t, int BLK_SIZE, int NPEL>
__device__ void dev_expanding_search_2(
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  int2 area[] = {
    { -2, -2 },
    { -1, -2 },
    { 0, -2 },
    { 1, -2 },
    { 2, -2 },

    { -2, -1 },
    { 2, -1 },
    { -2, 0 },
    { 2, 0 },
    { -2, 1 },
    { 2, 1 },

    { -2, 2 },
    { -1, 2 },
    { 0, 2 },
    { 1, 2 },
    { 2, 2 }
  };

  __shared__ bool isVectorOK[16];
  __shared__ CostResult result[16];
  __shared__ const pixel_t* pRefY[16];
  __shared__ const pixel_t* pRefU[16];
  __shared__ const pixel_t* pRefV[16];

  if (tx < 16) {
    int x = result[tx].xy.x = cx + area[tx].x;
    int y = result[tx].xy.y = cy + area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }

    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x / 2, y / 2);
    pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x / 2, y / 2);
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += (sad * PENALTY_NEW) >> 8;
    }
  }
  int bx2 = bx + 8;
  if (isVectorOK[bx2]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx2], pRefU[bx2], pRefV[bx2], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx2].cost += (sad * PENALTY_NEW) >> 8;
    }
  }

  __syncthreads();

  // 結果集約
  if (tx < 8) { // reduceは16-8=8スレッドで呼ぶ
    dev_reduce_result<16>(result, tx);

    if (tx == 0) { // tx == 0は最後のデータを書き込んでいるのでアクセスOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

// __syncthreads()を呼び出しているので全員で呼ぶ
template <typename pixel_t, int BLK_SIZE, int NPEL>
__device__ void dev_hex2_search_1(
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  int2 area[] = { { -1,-2 },{ -2,0 },{ -1,2 },{ 1,2 },{ 2,0 },{ 1,-2 },{ -1,-2 },{ -2,0 } };

  __shared__ bool isVectorOK[8];
  __shared__ CostResult result[8];
  __shared__ const pixel_t* pRefY[8];
  __shared__ const pixel_t* pRefU[8];
  __shared__ const pixel_t* pRefV[8];

  isVectorOK[tx] = false;

  if (tx < 6) {
    int x = result[tx].xy.x = cx + area[tx].x;
    int y = result[tx].xy.y = cy + area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }

    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x / 2, y / 2);
    pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x / 2, y / 2);
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += (sad * PENALTY_NEW) >> 8;
    }
  }

  __syncthreads();

  // 結果集約
  if (tx < 2) { // reduceは6-4=2スレッドで呼ぶ
    dev_reduce_result<8>(result, tx);

    if (tx == 0) { // tx == 0は最後のデータを書き込んでいるのでアクセスOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

// SRCH_DIMX % BLK_SIZE == 0が条件
template <typename pixel_t, int BLK_SIZE>
__device__ void dev_read_pixels(int tx, const pixel_t* src, int nPitch, int offx, int offy, pixel_t *dst)
{
  int y = tx / BLK_SIZE;
  int x = tx % BLK_SIZE;
  if (BLK_SIZE == 8) {
    if (y < 8) {
      dst[x + y * BLK_SIZE] = src[(x + offx) + (y + offy) * nPitch];
    }
  }
  else if (BLK_SIZE == 16) {
    dst[x + y * BLK_SIZE] = src[(x + offx) + (y + offy) * nPitch];
    y += 8;
    dst[x + y * BLK_SIZE] = src[(x + offx) + (y + offy) * nPitch];
  }
  else if (BLK_SIZE == 32) {
    for (; y < BLK_SIZE; y += SRCH_DIMX / BLK_SIZE) {
      dst[x + y * BLK_SIZE] = src[(x + offx) + (y + offy) * nPitch];
    }
  }
}

template <typename pixel_t, int BLK_SIZE, int SEARCH, int NPEL>
__global__ void kl_search(
  int nBlkX, int nBlkY, const SearchBlock* __restrict__ blocks,
  short2* vectors, // [x,y]
  int nPad,
  const pixel_t* __restrict__ pSrcY, const pixel_t* __restrict__ pSrcU, const pixel_t* __restrict__ pSrcV,
  const pixel_t* __restrict__ pRefY, const pixel_t* __restrict__ pRefU, const pixel_t* __restrict__ pRefV,
  int nPitchY, int nPitchUV,
  int nImgPitchY, int nImgPitchUV
)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    BLK_STEP = BLK_SIZE / 2,
  };

  const int tx = threadIdx.x;
  const int wi = tx % 16;
  const int bx = tx / 16;

  for (int blkx = blockIdx.x; blkx < nBlkX; blkx += blockDim.x) {
    for (int blky = 0; blky < nBlkY; ++blky) {

      // srcをshared memoryに転送
      int offx = nPad + blkx * BLK_STEP;
      int offy = nPad + blky * BLK_STEP;

      __shared__ pixel_t srcY[BLK_SIZE * BLK_SIZE];
      __shared__ pixel_t srcU[BLK_SIZE_UV * BLK_SIZE_UV];
      __shared__ pixel_t srcV[BLK_SIZE_UV * BLK_SIZE_UV];

      dev_read_pixels<pixel_t, BLK_SIZE>(tx, pSrcY, nPitchY, offx, offy, srcY);
      dev_read_pixels<pixel_t, BLK_SIZE_UV>(tx, pSrcU, nPitchUV, offx / 2, offy / 2, srcU);
      dev_read_pixels<pixel_t, BLK_SIZE_UV>(tx, pSrcV, nPitchUV, offx / 2, offy / 2, srcV);

      __shared__ const pixel_t* pRefBY;
      __shared__ const pixel_t* pRefBU;
      __shared__ const pixel_t* pRefBV;

      if (tx == 0) {
        pRefBY = &pRefY[offx + offy * nPitchY];
        pRefBU = &pRefU[offx / 2 + offy / 2 * nPitchU];
        pRefBV = &pRefV[offx / 2 + offy / 2 * nPitchV];
      }

      // パラメータなどのデータをshared memoryに格納
      __shared__ int data[12];
      __shared__ sad_t dataf[5];

      if (tx < 12) {
        int blkIdx = blky*nBlkX + blkx;
        data[tx] = blocks[blkIdx].data[tx];
        if (tx < 5) {
          dataf[tx] = blocks[blkIdx].dataf[tx];
        }
      }

      __syncthreads();

      // FetchPredictors
      __shared__ CostResult result[8];
      __shared__ const pixel_t* pRefY[8];
      __shared__ const pixel_t* pRefU[8];
      __shared__ const pixel_t* pRefV[8];

      if (tx < 6) {
        __shared__ volatile short pred[7][2]; // x, y

        // zero, global, predictor, predictors[1]～[3]を取得
        short2 vec = vectors[REF_VECTOR_INDEX[tx]];
        dev_clip_mv(vec, CLIP_RECT);
        pred[tx][0] = vec.x;
        pred[tx][1] = vec.y;
        // memfence
        if (tx < 2) {
          // Median predictor
          // 計算効率が悪いので消したい・・・
          int a = pred[3][tx];
          int b = pred[4][tx];
          int c = pred[5][tx];
          int max_ = dev_max(a, b, c);
          int min_ = dev_min(a, b, c);
          int med_ = a + b + c - max_ - min_;
          pred[6][tx] = med_;
        }
        // memfence
        int x = result[tx].xy.x = pred[tx][0];
        int y = result[tx].xy.y = pred[tx][1];
        result[tx].cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

        pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
        pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchUV, nImgPitchUV, x / 2, y / 2);
        pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchUV, nImgPitchUV, x / 2, y / 2);
      }

      __syncthreads();

      // まずは7箇所を計算
      if (bx < 7) {
        sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE>(wi, srcY, srcU, srcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchUV, nPitchUV);
        if (wi == 0) {
          if (bx < 3) {
            // pzero, pglobal, 1
            result[bx].cost = (sad * PENALTIES[bx]) >> 8;
          }
          else {
            result[bx].cost += sad;
          }
        }
        // とりあえず比較はcostだけでやるのでSADは要らない
        // SADは探索が終わったら再計算する
      }

      __syncthreads();

      // 結果集約
      if (tx < 3) { // 7-4=3スレッドで呼ぶ
        dev_reduce_result<8>(result, tx);
      }

      __syncthreads();

      // Refine
      if (SEARCH == 1) {
        // EXHAUSTIVE
        int bmx = result[0].xy.x;
        int bmy = result[0].xy.y;
        dev_expanding_search_1<pixel_t, BLK_SIZE, NPEL>(
          tx, wi, bx, bmx, bmy, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
        dev_expanding_search_2<pixel_t, BLK_SIZE, NPEL>(
          tx, wi, bx, bmx, bmy, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
      }
      else if (SEARCH == 2) {
        // HEX2SEARCH
        dev_hex2_search_1<pixel_t, BLK_SIZE, NPEL>(
          tx, wi, bx, result[0].xy.x, result[0].xy.y, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
        dev_expanding_search_1<pixel_t, BLK_SIZE, NPEL>(
          tx, wi, bx, result[0].xy.x, result[0].xy.y, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
      }

      // 結果書き込み
      if (tx == 0) {
        vectors[blky*nBlkX + blkx] = result[0].xy;
      }

      // 共有メモリ保護
      __syncthreads();
    }
  }
}

// threads=128,
template <typename pixel_t, int BLK_SIZE, int NPEL>
__global__ void kl_calc_all_sad(
  int nBlkX, int nBlkY,
  const short2* vectors, // [x,y]
  int* dst_sad,
  int nPad,
  const pixel_t* __restrict__ pSrcY, const pixel_t* __restrict__ pSrcU, const pixel_t* __restrict__ pSrcV,
  const pixel_t* __restrict__ pRefY, const pixel_t* __restrict__ pRefU, const pixel_t* __restrict__ pRefV,
  int nPitchY, int nPitchUV,
  int nImgPitchY, int nImgPitchUV)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    BLK_STEP = BLK_SIZE / 2,
  };

  int tid = threadIdx.x;
  int x = blockIdx.x * blockDim.x;
  int y = blockIdx.y * blockDim.y;

  int offx = nPad + blkx * BLK_STEP;
  int offy = nPad + blky * BLK_STEP;

  __shared__ const pixel_t* pRefBY;
  __shared__ const pixel_t* pRefBU;
  __shared__ const pixel_t* pRefBV;
  __shared__ const pixel_t* pSrcBY;
  __shared__ const pixel_t* pSrcBU;
  __shared__ const pixel_t* pSrcBV;

  if (tid == 0) {
    pRefBY = &pRefY[offx + offy * nPitchY];
    pRefBU = &pRefU[offx / 2 + offy / 2 * nPitchU];
    pRefBV = &pRefV[offx / 2 + offy / 2 * nPitchV];
    pSrcBY = &pSrcY[offx + offy * nPitchY];
    pSrcBU = &pSrcU[offx / 2 + offy / 2 * nPitchU];
    pSrcBV = &pSrcV[offx / 2 + offy / 2 * nPitchV];

    short2 xy = vectors[x + y * nBlkX];

    pRefBY = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, xy.x, xy.y);
    pRefBU = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchUV, nImgPitchUV, xy.x / 2, xy.y / 2);
    pRefBV = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchUV, nImgPitchUV, xy.x / 2, xy.y / 2);
  }

  __syncthreads();

  int sad = 0;
  if (BLK_SIZE == 16) {
    // 16x16
    int yx = tid % 16;
    int yy = tid / 16;
    for (int t = 0; t < 2; ++t, yy += 8) { // 2回ループ
      sad = __sad(pSrcBY[yx + yy * nPitchY], pRefBY[yx + yy * nPitchY], sad);
    }
    // UVは8x8
    int uvx = tid % 8;
    int uvy = tid / 8;
    if (uvy > 8) {
      uvy -= 8;
      sad = __sad(pSrcBU[uvx + uvy * nPitchUV], pRefBU[uvx + uvy * nPitchUV], sad);
    }
    else {
      sad = __sad(pSrcBV[uvx + uvy * nPitchUV], pRefBV[uvx + uvy * nPitchUV], sad);
    }
  }
  else if (BLK_SIZE == 32) {
    // 32x32
    int yx = tid % 32;
    int yy = tid / 32;
    for (int t = 0; t < 8; ++t, yy += 4) { // 8回ループ
      sad = __sad(pSrcBY[yx + yy * nPitchY], pRefBY[yx + yy * nPitchY], sad);
    }
    // ブロックサイズがスレッド数と一致
    int uvx = tid % 16;
    int uvy = tid / 16;
    for (int t = 0; t < 2; ++t, uvy += 8) { // 2回ループ
      sad = __sad(pSrcBU[uvx + uvy * nPitchUV], pRefBU[uvx + uvy * nPitchUV], sad);
      sad = __sad(pSrcBV[uvx + uvy * nPitchUV], pRefBV[uvx + uvy * nPitchUV], sad);
    }
  }
  sad = dev_reduce<128>(sad, tid);
  
  if (tx == 0) {
    dst_sad[x + y * nBlkX] = sad;
  }
}

__global__ void kl_prepare_search(
  int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
  sad_t nCurrentLambda, sad_t lsad,
  sad_t penaltyZero, sad_t penaltyGlobal, sad_t penaltyNew,
  int nPel, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
  const int* sads, SearchBlock* dst_blocks)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nBlkX && y < nBlkY) {
    //
    int blkIdx = x + y*nBlkX;
    int sad = sads[blkIdx];
    SearchBlock *data = &dst_blocks[blkIdx];

    int x = nPad + nBlkSizeOvr * x;
    int y = nPad + nBlkSizeOvr * y;
    //
    int nPaddingScaled = nPad >> nLogScale;

    int nDxMax = nPel * (nExtendedWidth - x - nBlkSize - nPad + nPaddingScaled) - 1;
    int nDyMax = nPel * (nExptendedHeight - y - nBlkSize - nPad + nPaddingScaled) - 1;
    int nDxMin = -nPel * (x - nPad + nPaddingScaled);
    int nDyMin = -nPel * (y - nPad + nPaddingScaled);

    data->data[0] = nDxMax;
    data->data[1] = nDyMax;
    data->data[2] = nDxMin;
    data->data[3] = nDyMin;

    int p1 = -2; // -2はzeroベクタ
    // Left (or right) predictor
    if (x >= 2)
    {
      p1 = blkIdx - (1 + (x & 1));
    }

    int p2 = -2;
    // Up predictor
    if (y > 0)
    {
      p2 = blkIdx - nBlkX;
    }
    else {
      // medianでleftを選ばせるため
      p2 = p1;
    }

    int p3 = -2;
    // bottom-right pridictor (from coarse level)
    if ((y < nBlkY - 1) && (x < nBlkX - 1))
    {
      p3 = blkIdx + nBlkX + 1;
    }

    data->data[4] = -2;    // zero
    data->data[5] = -1;    // global
    data->data[6] = blkIdx;// predictor
    data->data[7] = p1;    //  predictors[1]
    data->data[8] = p2;    //  predictors[2]
    data->data[9] = p3;    //  predictors[3]

    data->dataf[0] = penaltyZero;
    data->dataf[1] = penaltyGlobal;
    data->dataf[2] = 1;
    data->dataf[3] = penaltyNew;

    sad_t lambda = nCurrentLambda * lsad / (lsad + (sad >> 1)) * lsad / (lsad + (sad >> 1));
    data->dataf[4] = lambda;
  }
}

// threads=(1024), blocks=(2)
__global__ void kl_most_freq_mv(short2* vectors, int nVec, short2* globalMVec)
{
  enum {
    DIMX = 1024,
    // level==1が最大なので、このサイズで8Kくらいまで対応
    FREQ_SIZE = DIMX*8,
    HALF_SIZE = FREQ_SIZE / 2
  };

  int tid = threadIdx.x;

  union SharedBuffer {
    int freq_arr[FREQ_SIZE]; //32KB
    struct {
      int red_cnt[DIMX];
      int red_idx[DIMX];
    };
  };
  __shared__ SharedBuffer b;

  for (int i = 0; i < FREQ_SIZE/ DIMX; i += DIMX) {
    b.freq_arr[tid + i * DIMX] = 0;
  }
  __syncthreads();

  if (blockIdx.x == 0) {
    // x
    for (int i = tid; i < nVec; i += DIMX) {
      atomicAdd(&b.freq_arr[vectors[i].x + HALF_SIZE], 1);
    }
  }
  else {
    // y
    for (int i = tid; i < nVec; i += DIMX) {
      atomicAdd(&b.freq_arr[vectors[i].y + HALF_SIZE], 1);
    }
  }
  __syncthreads();

  int maxcnt = 0;
  int index = 0;
  for (int i = 0; i < FREQ_SIZE / DIMX; i += DIMX) {
    if (b.freq_arr[tid + i * DIMX] > maxcnt) {
      maxcnt = b.freq_arr[tid + i * DIMX];
      index = tid + i * DIMX;
    }
  }
  __syncthreads();

  dev_reduce2<int, int, DIMX, CountIndexReducer>(tid, maxcnt, index, b.red_cnt, b.red_idx);

  if (tid == 0) {
    if (blockIdx.x == 0) {
      // x
      globalMVec->x = index - HALF_SIZE;
    }
    else {
      // y
      globalMVec->y = index - HALF_SIZE;
    }
  }
}

__global__ void kl_mean_global_mv(short2* vectors, int nVec, short2* globalMVec)
{
  enum {
    DIMX = 1024,
  };

  int tid = threadIdx.x;
  int medianx = globalMVec->x;
  int mediany = globalMVec->y;

  int meanvx = 0;
  int meanvy = 0;
  int num = 0;

  for (int i = tid; i < nVec; i += DIMX) {
    if (__sad(vectors[i].x, medianx, 0) < 6
      && __sad(vectors[i].y, mediany, 0) < 6)
    {
      meanvx += vectors[i].x;
      meanvy += vectors[i].y;
      num += 1;
    }
  }

  __shared__ int red_vx[DIMX];
  __shared__ int red_vy[DIMX];
  __shared__ int red_num[DIMX];

  red_vx[tid] = meanvx;
  red_vy[tid] = meanvy;
  red_num[tid] = num;

  __syncthreads();
  if (tid < 512) {
    red_vx[tid] += red_vx[tid + 512];
    red_vy[tid] += red_vy[tid + 512];
    red_num[tid] += red_num[tid + 512];
  }
  __syncthreads();
  if (tid < 256) {
    red_vx[tid] += red_vx[tid + 256];
    red_vy[tid] += red_vy[tid + 256];
    red_num[tid] += red_num[tid + 256];
  }
  __syncthreads();
  if (tid < 128) {
    red_vx[tid] += red_vx[tid + 128];
    red_vy[tid] += red_vy[tid + 128];
    red_num[tid] += red_num[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    red_vx[tid] += red_vx[tid + 64];
    red_vy[tid] += red_vy[tid + 64];
    red_num[tid] += red_num[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    red_vx[tid] += red_vx[tid + 32];
    red_vy[tid] += red_vy[tid + 32];
    red_num[tid] += red_num[tid + 32];
  }
  __syncthreads();
  meanvx = red_vx[tid];
  meanvy = red_vy[tid];
  num = red_num[tid];
  if (tid < 32) {
    meanvx += __shfl_down(meanvx, 16);
    meanvy += __shfl_down(meanvy, 16);
    num += __shfl_down(num, 16);
    meanvx += __shfl_down(meanvx, 8);
    meanvy += __shfl_down(meanvy, 8);
    num += __shfl_down(num, 8);
    meanvx += __shfl_down(meanvx, 4);
    meanvy += __shfl_down(meanvy, 4);
    num += __shfl_down(num, 4);
    meanvx += __shfl_down(meanvx, 2);
    meanvy += __shfl_down(meanvy, 2);
    num += __shfl_down(num, 2);
    meanvx += __shfl_down(meanvx, 1);
    meanvy += __shfl_down(meanvy, 1);
    num += __shfl_down(num, 1);

    if (tid == 0) {
      globalMVec->x = 2 * meanvx / num;
      globalMVec->y = 2 * meanvy / num;
    }
  }
}

// normFactor = 3 - nLogPel + pob.nLogPel
// normov = (nBlkSizeX - nOverlapX)*(nBlkSizeY - nOverlapY)
// aoddx = (nBlkSizeX * 3 - nOverlapX * 2)
// aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
// aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
// aeveny = (nBlkSizeY * 3 - nOverlapY * 4);
// atotalx = (nBlkSizeX - nOverlapX) * 4
// atotaly = (nBlkSizeY - nOverlapY) * 4
__global__ void kl_interpolate_prediction(
  const short2* src_vector, const int* src_sad,
  short2* dst_vector, int* dst_sad,
  int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
  int normFactor, int normov, int atotal, int aodd, int aeven)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nDstBlkX && y < nDstBlkY) {
    short2 v1, v2, v3, v4;
    int sad1, sad2, sad3, sad4;
    int i = x;
    int j = y;
    if (i >= 2 * nSrcBlkX)
    {
      i = 2 * nSrcBlkX - 1;
    }
    if (j >= 2 * nSrcBlkY)
    {
      j = 2 * nSrcBlkY - 1;
    }
    int offy = -1 + 2 * (j % 2);
    int offx = -1 + 2 * (i % 2);
    int iper2 = i / 2;
    int jper2 = j / 2;

    if ((i == 0) || (i >= 2 * nSrcBlkX - 1))
    {
      if ((j == 0) || (j >= 2 * nSrcBlkY - 1))
      {
        v1 = v2 = v3 = v4 = src_vector[iper2 + (jper2)* nSrcBlkX];
        sad1 = sad2 = sad3 = sad4 = src_sad[iper2 + (jper2)* nSrcBlkX];
      }
      else
      {
        v1 = v2 = src_vector[iper2 + (jper2)* nSrcBlkX];
        sad1 = sad2 = src_sad[iper2 + (jper2)* nSrcBlkX];
        v3 = v4 = src_vector[iper2 + (jper2 + offy) * nSrcBlkX];
        sad3 = sad4 = src_sad[iper2 + (jper2 + offy) * nSrcBlkX];
      }
    }
    else if ((j == 0) || (j >= 2 * nSrcBlkY - 1))
    {
      v1 = v2 = src_vector[iper2 + (jper2)* nSrcBlkX];
      sad1 = sad2 = src_sad[iper2 + (jper2)* nSrcBlkX];
      v3 = v4 = src_vector[iper2 + offx + (jper2)* nSrcBlkX];
      sad3 = sad4 = src_sad[iper2 + offx + (jper2)* nSrcBlkX];
    }
    else
    {
      v1 = src_vector[iper2 + (jper2)* nSrcBlkX];
      sad1 = src_sad[iper2 + (jper2)* nSrcBlkX];
      v2 = src_vector[iper2 + offx + (jper2)* nSrcBlkX];
      sad2 = src_sad[iper2 + offx + (jper2)* nSrcBlkX];
      v3 = src_vector[iper2 + (jper2 + offy) * nSrcBlkX];
      sad3 = src_sad[iper2 + (jper2 + offy) * nSrcBlkX];
      v4 = src_vector[iper2 + offx + (jper2 + offy) * nSrcBlkX];
      sad4 = src_sad[iper2 + offx + (jper2 + offy) * nSrcBlkX];
    }

    int	ax1 = (offx > 0) ? aodd : aeven;
    int ax2 = atotal - ax1;
    int ay1 = (offy > 0) ? aodd : aeven;
    int ay2 = atotal - ay1;
    int a11 = ax1*ay1, a12 = ax1*ay2, a21 = ax2*ay1, a22 = ax2*ay2;
    int vx = (a11*v1.x + a21*v2.x + a12*v3.x + a22*v4.x) / normov;
    int vy = (a11*v1.y + a21*v2.y + a12*v3.y + a22*v4.y) / normov;
    
    sad_t tmp_sad = ((sad_t)a11*sad1 + (sad_t)a21*sad2 + (sad_t)a12*sad3 + (sad_t)a22*sad4) / normov;

    if (normFactor > 0) {
      vx >>= normFactor;
      vy >>= normFactor;
    }
    else {
      vx <<= -normFactor;
      vy <<= -normFactor;
    }

    int index = x + y * nDstBlkX;
    short2 v = { vx,vy };
    dst_vector[index] = v;
    dst_sad[index] = (int)(tmp_sad >> 4);
  }
}

__global__ void kl_load_mv(
	const VECTOR* in,
	short2* vectors, // [x,y]
	int* sads,
	int nBlk)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < nBlk) {
		VECTOR vin = in[x];
		short2 v = { vin.x, vin, y };
		vectors[x] = v;
		sads[x] = vin.sad;
	}
}

__global__ void kl_store_mv(
	VECTOR* dst,
	short2* vectors, // [x,y]
	int* sads,
	int nBlk)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < nBlk) {
		short2 v = vectors[x];
		VECTOR vout = { v.x, v.y, sads[x] };
		vectors[x] = vout;
	}
}

__global__ void kl_write_default_mv(VECTOR* dst, int nBlkCount, int verybigSAD)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x < nBlkCount) {
		dst[x].x = 0;
		dst[x].y = 0;
		dst[x].x = verybigSAD;
  }
}


