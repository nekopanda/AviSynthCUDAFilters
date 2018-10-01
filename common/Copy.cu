#include "Copy.h"
#include "CommonFunctions.h"
#include <string.h>

template <typename pixel_t>
__global__ void kl_copy(
  pixel_t* __restrict__ dst, int dst_pitch, const pixel_t* __restrict__ src, int src_pitch, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * dst_pitch] = src[x + y * src_pitch];
  }
}

void Copy_(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height, PNeoEnv env)
{
  if (src_pitch == 0) return;

  if (IS_CUDA) {
    if (((uintptr_t)dstp | (uintptr_t)srcp | dst_pitch | src_pitch) & 3) {
      // alignment‚È‚µ
      dim3 threads(32, 8);
      dim3 blocks(nblocks(row_size, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> > (dstp, dst_pitch, srcp, src_pitch, row_size, height);
      DEBUG_SYNC;
    }
    else if (((uintptr_t)dstp | (uintptr_t)srcp | dst_pitch | src_pitch) & 15) {
      // 4 byte align
      int width4 = (row_size + 3) >> 2;
      int dst_pitch4 = dst_pitch >> 2;
      int src_pitch4 = src_pitch >> 2;
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> > ((int*)dstp, dst_pitch4, (const int*)srcp, src_pitch4, width4, height);
      DEBUG_SYNC;
    }
    else {
      // 16 byte align
      int width4 = (row_size + 15) >> 4;
      int dst_pitch4 = dst_pitch >> 4;
      int src_pitch4 = src_pitch >> 4;
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> > ((int4*)dstp, dst_pitch4, (const int4*)srcp, src_pitch4, width4, height);
      DEBUG_SYNC;
    }
  }
  else {
    env->BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
  }
}
