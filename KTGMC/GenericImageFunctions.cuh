#pragma once


/////////////////////////////////////////////////////////////////////////////
// Elementwise
/////////////////////////////////////////////////////////////////////////////

template <typename pixel_t, typename F>
__global__ void kl_elementwise(
  pixel_t* dst, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    F f;
    f(dst[x + y * pitch]);
  }
}

template <typename pixel_t, typename F>
void launch_elementwise(
  pixel_t* dst, int width, int height, int pitch)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_elementwise<pixel_t, F> << <blocks, threads >> > (
    dst, width, height, pitch);
}

template <typename d_type, typename s_type, typename F>
__global__ void kl_elementwise(
  d_type* dst, const s_type* src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    F f;
    f(dst[x + y * pitch], src[x + y * pitch]);
  }
}

template <typename d_type, typename s_type, typename F>
void launch_elementwise(
  d_type* dst, const s_type* src, int width, int height, int pitch)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_elementwise<d_type, s_type, F> << <blocks, threads >> > (
    dst, src, width, height, pitch);
}

template <typename pixel_t, typename F>
__global__ void kl_elementwise(
  pixel_t* dst, const pixel_t* src0, const pixel_t* src1, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    F f;
    f(dst[x + y * pitch], src0[x + y * pitch], src1[x + y * pitch]);
  }
}

template <typename pixel_t, typename F>
void launch_elementwise(
  pixel_t* dst, const pixel_t* src0, const pixel_t* src1, int width, int height, int pitch)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_elementwise<pixel_t, F> << <blocks, threads >> > (
    dst, src0, src1, width, height, pitch);
}

template <typename T>
struct SetZeroFunction {
  __device__ void operator()(T& a) {
    a = VHelper<T>::make(0);
  }
};

template <typename T>
struct CopyFunction {
  __device__ void operator()(T& a, T b) {
    a = b;
  }
};
