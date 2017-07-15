#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "KDeintKernel.h"

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
