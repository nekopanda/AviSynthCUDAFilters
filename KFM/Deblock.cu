
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "Copy.h"

#include "VectorFunctions.cuh"

typedef uint8_t pixel_t;
typedef uchar4 vpixel_t;

// got from the following command in python. (do "from math import *" before)
const float S1 = 0.19509032201612825f;   // sin(1*pi/(2*8))
const float C1 = 0.9807852804032304f;    // cos(1*pi/(2*8))
const float S3 = 0.5555702330196022f;    // sin(3*pi/(2*8))
const float C3 = 0.8314696123025452f;    // cos(3*pi/(2*8))
const float S2S6 = 1.3065629648763766f;  // sqrt(2)*sin(6*pi/(2*8))
const float S2C6 = 0.5411961001461971f;  // sqrt(2)*cos(6*pi/(2*8))
const float S2 = 1.4142135623730951f;    // sqrt(2)

template <int stride>
__device__ void dev_dct8(float* data)
{
  // stage 1
  float a0 = data[7 * stride] + data[0 * stride];
  float a1 = data[6 * stride] + data[1 * stride];
  float a2 = data[5 * stride] + data[2 * stride];
  float a3 = data[4 * stride] + data[3 * stride];
  float a4 = data[3 * stride] - data[4 * stride];
  float a5 = data[2 * stride] - data[5 * stride];
  float a6 = data[1 * stride] - data[6 * stride];
  float a7 = data[0 * stride] - data[7 * stride];

  // stage 2 even
  float b0 = a3 + a0;
  float b1 = a2 + a1;
  float b2 = a1 - a2;
  float b3 = a0 - a3;

  // stage 2 odd
  float b4 = (S3 - C3) * a7 + C3 * (a4 + a7);
  float b5 = (S1 - C1) * a6 + C1 * (a5 + a6);
  float b6 = -(C1 + S1) * a5 + C1 * (a5 + a6);
  float b7 = -(C3 + S3) * a4 + C3 * (a4 + a7);

  // stage3 even
  float c0 = b1 + b0;
  float c1 = b0 - b1;
  float c2 = (S2S6 - S2C6) * b3 + S2C6 * (b2 + b3);
  float c3 = -(S2C6 + S2S6) * b2 + S2C6 * (b2 + b3);

  // stage3 odd
  float c4 = b6 + b4;
  float c5 = b7 - b5;
  float c6 = b4 - b6;
  float c7 = b5 + b7;

  // stage 4 odd
  float d4 = c7 - c4;
  float d5 = c5 * S2;
  float d6 = c6 * S2;
  float d7 = c4 + c7;

  // store
  data[0 * stride] = c0;
  data[4 * stride] = c1;
  data[2 * stride] = c2;
  data[6 * stride] = c3;
  data[7 * stride] = d4;
  data[3 * stride] = d5;
  data[5 * stride] = d6;
  data[1 * stride] = d7;
}

template <int stride>
__device__ void dev_idct8(float* data)
{
  float c0 = data[0 * stride];
  float c1 = data[4 * stride];
  float c2 = data[2 * stride];
  float c3 = data[6 * stride];
  float d4 = data[7 * stride];
  float d5 = data[3 * stride];
  float d6 = data[5 * stride];
  float d7 = data[1 * stride];

  float c4 = d7 - d4;
  float c5 = d5 * S2;
  float c6 = d6 * S2;
  float c7 = d4 + d7;

  float b0 = c1 + c0;
  float b1 = c0 - c1;
  float b2 = -(S2C6 + S2S6) * c3 + S2C6 * (c2 + c3);
  float b3 = (S2S6 - S2C6) * c2 + S2C6 * (c2 + c3);

  float b4 = c6 + c4;
  float b5 = c7 - c5;
  float b6 = c4 - c6;
  float b7 = c5 + c7;

  float a0 = b3 + b0;
  float a1 = b2 + b1;
  float a2 = b1 - b2;
  float a3 = b0 - b3;

  float a4 = -(C3 + S3) * b7 + C3 * (b4 + b7);
  float a5 = -(C1 + S1) * b6 + C1 * (b5 + b6);
  float a6 = (S1 - C1) * b5 + C1 * (b5 + b6);
  float a7 = (S3 - C3) * b4 + C3 * (b4 + b7);

  data[0 * stride] = a7 + a0;
  data[1 * stride] = a6 + a1;
  data[2 * stride] = a5 + a2;
  data[3 * stride] = a4 + a3;
  data[4 * stride] = a3 - a4;
  data[5 * stride] = a2 - a5;
  data[6 * stride] = a1 - a6;
  data[7 * stride] = a0 - a7;
}

__device__ void dev_dct8x8(int tx, float* data)
{
  dev_dct8<1>(data + tx * 8); // row
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
  dev_dct8<8>(data + tx);  // column
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
}

__device__ void dev_idct8x8(int tx, float* data)
{
  dev_idct8<1>(data + tx * 8); // row
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
  dev_idct8<8>(data + tx);  // column
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif
}

__device__ void dev_hardthresh(int tx, float *data, int qp, float strength)
{
  float threshold = qp * ((1 << 4) + strength) - 1;

  for (int i = tx; i < 64; i += 8) {
    if (i == 0) continue;
    float level = data[i];
    if (abs(level) <= threshold) {
      data[i] = 0;
    }
  }
}

__device__ void dev_softthresh(int tx, float *data, int qp, float strength)
{
  float threshold = qp * ((1 << 4) + strength) - 1;

  for (int i = tx; i < 64; i += 8) {
    if (i == 0) continue;
    float level = data[i];
    if (abs(level) <= threshold) data[i] = 0;
    else if (level > 0) data[i] -= threshold;
    else                data[i] += threshold;
  }
}

__constant__ __host__ uchar2 g_deblock_offset[127] = {
  { 0,0 },
  { 0,0 },{ 4,4 },                                           // quality = 1
  { 0,0 },{ 2,2 },{ 6,4 },{ 4,6 },                             // quality = 2
  { 0,0 },{ 5,1 },{ 2,2 },{ 7,3 },{ 4,4 },{ 1,5 },{ 6,6 },{ 3,7 }, // quality = 3

  { 0,0 },{ 4,0 },{ 1,1 },{ 5,1 },{ 3,2 },{ 7,2 },{ 2,3 },{ 6,3 }, // quality = 4
  { 0,4 },{ 4,4 },{ 1,5 },{ 5,5 },{ 3,6 },{ 7,6 },{ 2,7 },{ 6,7 },

  { 0,0 },{ 0,2 },{ 0,4 },{ 0,6 },{ 1,1 },{ 1,3 },{ 1,5 },{ 1,7 }, // quality = 5
  { 2,0 },{ 2,2 },{ 2,4 },{ 2,6 },{ 3,1 },{ 3,3 },{ 3,5 },{ 3,7 },
  { 4,0 },{ 4,2 },{ 4,4 },{ 4,6 },{ 5,1 },{ 5,3 },{ 5,5 },{ 5,7 },
  { 6,0 },{ 6,2 },{ 6,4 },{ 6,6 },{ 7,1 },{ 7,3 },{ 7,5 },{ 7,7 },

  { 0,0 },{ 4,4 },{ 0,4 },{ 4,0 },{ 2,2 },{ 6,6 },{ 2,6 },{ 6,2 }, // quality = 6
  { 0,2 },{ 4,6 },{ 0,6 },{ 4,2 },{ 2,0 },{ 6,4 },{ 2,4 },{ 6,0 },
  { 1,1 },{ 5,5 },{ 1,5 },{ 5,1 },{ 3,3 },{ 7,7 },{ 3,7 },{ 7,3 },
  { 1,3 },{ 5,7 },{ 1,7 },{ 5,3 },{ 3,1 },{ 7,5 },{ 3,5 },{ 7,1 },
  { 0,1 },{ 4,5 },{ 0,5 },{ 4,1 },{ 2,3 },{ 6,7 },{ 2,7 },{ 6,3 },
  { 0,3 },{ 4,7 },{ 0,7 },{ 4,3 },{ 2,1 },{ 6,5 },{ 2,5 },{ 6,1 },
  { 1,0 },{ 5,4 },{ 1,4 },{ 5,0 },{ 3,2 },{ 7,6 },{ 3,6 },{ 7,2 },
  { 1,2 },{ 5,6 },{ 1,6 },{ 5,2 },{ 3,0 },{ 7,4 },{ 3,4 },{ 7,0 },
};

// src: 外周8ピクセル拡張したソース
// out: 外周8ピクセル拡張し、かつ、縦方向に4倍した中間出力バッファ
// qp_table: ffmpegから取得したqpテーブル
// offsets: ブロックオフセットテーブル
// shift: min(3, 16 - quality - bits)
// maxv: (1 << (11 - shift)) - 1
__global__ void kl_deblock(
  const pixel_t* src, int src_pitch,
  int bw, int bh,
  uint16_t* out, int out_pitch,
  const short* qp_table, int qp_pitch,
  int count_minus_1, int shift, int maxv,
  float strength, bool is_soft)
{
  int tx = threadIdx.x; // 8
  int ty = threadIdx.y; // count

  __shared__ int16_t local_out[16][16];
  extern __shared__ float dct_tmp_buf[];

  float* dct_tmp = dct_tmp_buf + 64 * ty;

  // local_out初期化
  for (int y = ty; y < 16; y += blockDim.y) {
    local_out[y][tx] = 0;
    local_out[y][tx + 8] = 0;
  }
  __syncthreads();

  // getpixel
  const char2 offset = g_deblock_offset[count_minus_1 + ty];
  int off_x = blockIdx.x * 8 + offset.x;
  int off_y = blockIdx.y * 8 + offset.y;
  for (int y = 0; y < 8; ++y) {
    dct_tmp[tx + y * 8] = src[(off_x + tx) + (off_y + y) * src_pitch];
  }

  // dct
  dev_dct8x8(tx, dct_tmp);

  // requantize
  int qp = qp_table[blockIdx.x + blockIdx.y * qp_pitch];
  if (is_soft) {
    dev_softthresh(tx, dct_tmp, qp, strength);
  }
  else {
    dev_hardthresh(tx, dct_tmp, qp, strength);
  }
#if CUDART_VERSION >= 9000
  __syncwarp();
#endif

  // idct
  dev_idct8x8(tx, dct_tmp);

  // add
  for (int y = 0; y < 8; ++y) {
    int tmp = clamp((int)(dct_tmp[tx + y * 8] + 4) >> shift, 0, maxv);
    atomicAdd(&local_out[offset.y + y][offset.x + tx], tmp);
  }
  __syncthreads();

  // store
  int off_z = (blockIdx.x & 1) + (blockIdx.y & 1) * 2;
  off_x = blockIdx.x * 8;
  off_y = (gridDim.y * off_z + blockIdx.y) * 8;
  for (int y = ty; y < 16; y += blockDim.y) {
    out[(off_x + 0) + (off_y + y) * out_pitch] = local_out[y][tx];
    out[(off_x + 8) + (off_y + y) * out_pitch] = local_out[y][tx + 8];
  }
}

void cpu_deblock(
  const pixel_t* src, int src_pitch,
  int bw, int bh,
  uint16_t* out, int out_pitch,
  const short* qp_table, int qp_pitch,
  int count_minus_1, int shift, int maxv,
  float strength, bool is_soft)
{
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      int16_t local_out[16][16];

      // local_out初期化
      memset(local_out, 0, sizeof(local_out));

      for (int ty = 0; ty <= count_minus_1; ++ty) {
        // getpixel

        // dct
        // requantize
        // idct
        // add
      }
      // store
    }
  }
}

// normalize the qscale factor
__device__ __host__ inline int norm_qscale(int qscale, int type)
{
  switch (type) {
  case 0/*FF_QSCALE_TYPE_MPEG1*/: return qscale;
  case 1/*FF_QSCALE_TYPE_MPEG2*/: return qscale >> 1;
  case 2/*FF_QSCALE_TYPE_H264*/:  return qscale >> 2;
  case 3/*FF_QSCALE_TYPE_VP56*/:  return (63 - qscale + 2) >> 2;
  }
  return qscale;
}


__global__ void kl_make_qp_table(
  int in_width, int in_height,
  const int8_t* in_table, int in_pitch, int qp_scale,
  int out_width, int out_height,
  short* out_table, int out_pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < out_width && y < out_height) {
    int qp;
    if (in_table) {
      qp = in_table[min(x, in_width - 1) + min(y, in_height - 1) * in_pitch];
      qp = max(1, norm_qscale(qp, qp_scale));
    }
    else {
      qp = qp_scale;
    }
    out_table[x + y * out_pitch] = qp;
  }
}

void cpu_make_qp_table(
  int in_width, int in_height,
  const int8_t* in_table, int in_pitch, int qp_scale,
  int out_width, int out_height,
  short* out_table, int out_pitch)
{
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      int qp;
      if (in_table) {
        qp = in_table[min(x, in_width - 1) + min(y, in_height - 1) * in_pitch];
        qp = max(1, norm_qscale(qp, qp_scale));
      }
      else {
        qp = qp_scale;
      }
      out_table[x + y * out_pitch] = qp;
    }
  }
}

__constant__ __host__ uchar4 g_ldither[8][2] = {
  { {  0,  48,  12,  60 }, {  3,  51,  15,  63 } },
  { { 32,  16,  44,  28 }, { 35,  19,  47,  31 } },
  { {  8,  56,   4,  52 }, { 11,  59,   7,  55 } },
  { { 40,  24,  36,  20 }, { 43,  27,  39,  23 } },
  { {  2,  50,  14,  62 }, {  1,  49,  13,  61 } },
  { { 34,  18,  46,  30 }, { 33,  17,  45,  29 } },
  { { 10,  58,   6,  54 }, {  9,  57,   5,  53 } },
  { { 42,  26,  38,  22 }, { 41,  25,  37,  21 } },
};

// shift: quality - (3 - deblock_shift)
__global__ void kl_merge_deblock(
  int width, int height,
  const short4 *tmp, int tmp_pitch,
  vpixel_t* out, int out_pitch, int shift, int maxv)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto sum = to_int(tmp[x + (height * 0 + y) * tmp_pitch]) +
      to_int(tmp[x + (height * 1 + y) * tmp_pitch]) +
      to_int(tmp[x + (height * 2 + y) * tmp_pitch]) +
      to_int(tmp[x + (height * 3 + y) * tmp_pitch]);
    auto tmp = to_float(sum) * (1.0f / (1 << shift)) +
      to_float(g_ldither[y & 3][x & 1]) * (1.0f / 64.0f);
    out[x + y * out_pitch] = VHelper<vpixel_t>::cast_to(clamp(tmp, 0, maxv));
  }
}

void cpu_merge_deblock(
  int width, int height,
  const short4 *tmp, int tmp_pitch,
  vpixel_t* out, int out_pitch, int shift, int maxv)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto sum = to_int(tmp[x + (height * 0 + y) * tmp_pitch]) +
        to_int(tmp[x + (height * 1 + y) * tmp_pitch]) +
        to_int(tmp[x + (height * 2 + y) * tmp_pitch]) +
        to_int(tmp[x + (height * 3 + y) * tmp_pitch]);
      auto tmp = to_float(sum) * (1.0f / (1 << shift)) +
        to_float(g_ldither[y & 3][x & 1]) * (1.0f / 64.0f);
      out[x + y * out_pitch] = VHelper<vpixel_t>::cast_to(clamp(tmp, 0, maxv));
    }
  }
}
