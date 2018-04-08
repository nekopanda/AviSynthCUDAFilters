#include "avisynth.h"
#include "avs/alignment.h"

#include "../AvsCUDA.h"

#include <stdint.h>
#include "CommonFunctions.h"
#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"

template <typename pixel_t>
__global__ void kl_copy(
  pixel_t* dst, int dst_pitch, const pixel_t* __restrict__ src, int src_pitch, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * dst_pitch] = src[x + y * src_pitch];
  }
}

void Copy(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height, PNeoEnv env)
{
  if (src_pitch == 0) return;

  if (IS_CUDA) {
    if (((uintptr_t)dstp | (uintptr_t)srcp | dst_pitch | src_pitch) & 3) {
      // alignment‚È‚µ
      dim3 threads(32, 8);
      dim3 blocks(nblocks(row_size, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> >(dstp, dst_pitch, srcp, src_pitch, row_size, height);
      DEBUG_SYNC;
    }
    else if (((uintptr_t)dstp | (uintptr_t)srcp | dst_pitch | src_pitch) & 15) {
      // 4 byte align
      int width4 = (row_size + 3) >> 2;
      int dst_pitch4 = dst_pitch >> 2;
      int src_pitch4 = src_pitch >> 2;
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> >((int*)dstp, dst_pitch4, (const int*)srcp, src_pitch4, width4, height);
      DEBUG_SYNC;
    }
    else {
      // 16 byte align
      int width4 = (row_size + 15) >> 4;
      int dst_pitch4 = dst_pitch >> 4;
      int src_pitch4 = src_pitch >> 4;
      dim3 threads(32, 8);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
      kl_copy << <blocks, threads >> >((int4*)dstp, dst_pitch4, (const int4*)srcp, src_pitch4, width4, height);
      DEBUG_SYNC;
    }
  }
  else {
    env->BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
  }
}


class Align : public GenericVideoFilter
{
  int systemFrameAlign;
  int isRGB;

  const int* GetPlanes() {
    static const int planesYUV[] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    static const int planesRGB[] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    return isRGB ? planesRGB : planesYUV;
  }

  template <typename T> T align_size(T v) {
    return (v + systemFrameAlign - 1) & ~(systemFrameAlign - 1);
  }

  template <typename pixel_t>
  void Proc(PVideoFrame& dst, PVideoFrame& src, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const int *planes = GetPlanes();

    for (int p = 0; p < 4; ++p) {
      int plane = planes[p];
      if (src->GetPitch(plane) == 0) continue;

      const uint8_t* pSrc = src->GetReadPtr(plane);
      uint8_t* pDst = dst->GetWritePtr(plane);
      int srcPitch = src->GetPitch(plane);
      int dstPitch = dst->GetPitch(plane);
      int rowSize = src->GetRowSize(plane);
      int height = src->GetHeight(plane);

      Copy(pDst, dstPitch, pSrc, srcPitch, rowSize, height, env);
    }
  }

  bool IsAligned(const PVideoFrame& frame)
  {
    const int *planes = GetPlanes();
    for (int p = 0; p < 4; ++p) {
      int plane = planes[p];
      if (frame->GetPitch(plane) == 0) continue;
      const BYTE* ptr = frame->GetReadPtr(plane);
      int pitch = frame->GetPitch(plane);
      int rowSize = frame->GetRowSize(plane);

      const BYTE* alignedPtr = (const BYTE*)align_size((uintptr_t)ptr);
      int alignedRowSize = align_size(rowSize);

      if (alignedPtr != ptr) return false;
      if (alignedRowSize != pitch) return false;
    }
    return true;
  }

public:
  Align(PClip child, IScriptEnvironment* env_)
    : GenericVideoFilter(child)
    , isRGB(vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    PNeoEnv env = env_;
    systemFrameAlign = env->GetProperty(AEP_FRAME_ALIGN);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    PVideoFrame src = child->GetFrame(n, env);
    if (IsAligned(src)) {
      // Šù‚ÉAlign‚³‚ê‚Ä‚¢‚é
      return src;
    }

    PVideoFrame dst = env->NewVideoFrame(vi);

    // for debug
    if (IsAligned(dst) == false) {
      env->ThrowError("[Align]: New allocated frame is not aligned !!!");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      Proc<uint8_t>(dst, src, env);
      break;
    case 2:
      Proc<uint16_t>(dst, src, env);
      break;
    case 4:
      Proc<float>(dst, src, env);
      break;
    default:
      env->ThrowError("[Align] Unsupported pixel format");
    }

    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    switch (cachehints) {
    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;
    case CACHE_GET_DEV_TYPE:
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new Align(
      args[0].AsClip(),
      env);
  }
};


/**************************************
********   DoubleWeaveFields   *******
*************************************/

void copy_field(const PVideoFrame& dst, const PVideoFrame& src, bool yuv, bool planarRGB, bool parity, IScriptEnvironment* env_)
{
  PNeoEnv env = env_;

  bool noTopBottom = yuv || planarRGB;

  int plane1 = planarRGB ? PLANAR_B : PLANAR_U;
  int plane2 = planarRGB ? PLANAR_R : PLANAR_V;

  const int add_pitch = dst->GetPitch() * (parity ^ noTopBottom);
  const int add_pitchUV = dst->GetPitch(plane1) * (parity ^ noTopBottom);
  const int add_pitchA = dst->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  Copy(dst->GetWritePtr() + add_pitch, dst->GetPitch() * 2,
    src->GetReadPtr(), src->GetPitch(),
    src->GetRowSize(), src->GetHeight(), env);

  Copy(dst->GetWritePtr(plane1) + add_pitchUV, dst->GetPitch(plane1) * 2,
    src->GetReadPtr(plane1), src->GetPitch(plane1),
    src->GetRowSize(plane1), src->GetHeight(plane1), env);

  Copy(dst->GetWritePtr(plane2) + add_pitchUV, dst->GetPitch(plane2) * 2,
    src->GetReadPtr(plane2), src->GetPitch(plane2),
    src->GetRowSize(plane2), src->GetHeight(plane2), env);

  Copy(dst->GetWritePtr(PLANAR_A) + add_pitchA, dst->GetPitch(PLANAR_A) * 2,
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A),
    src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A), env);
}

class DoubleWeaveFields : public GenericVideoFilter
{
public:
  DoubleWeaveFields(PClip _child)
    : GenericVideoFilter(_child)
  {
    vi.height *= 2;
    vi.SetFieldBased(false);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame a = child->GetFrame(n, env);
    PVideoFrame b = child->GetFrame(n + 1, env);

    PVideoFrame result = env->NewVideoFrame(vi);

    const bool parity = child->GetParity(n);

    copy_field(result, a, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
    copy_field(result, b, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);

    return result;
  }


  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    switch (cachehints) {
    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;
    case CACHE_GET_DEV_TYPE:
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };
};


/**************************************
********   DoubleWeaveFrames   *******
*************************************/

void copy_alternate_lines(const PVideoFrame& dst, const PVideoFrame& src, bool yuv, bool planarRGB, bool parity, IScriptEnvironment* env_)
{
  PNeoEnv env = env_;

  bool noTopBottom = yuv || planarRGB;

  int plane1 = planarRGB ? PLANAR_B : PLANAR_U;
  int plane2 = planarRGB ? PLANAR_R : PLANAR_V;

  const int src_add_pitch = src->GetPitch()         * (parity ^ noTopBottom);
  const int src_add_pitchUV = src->GetPitch(plane1) * (parity ^ noTopBottom);
  const int src_add_pitchA = src->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  const int dst_add_pitch = dst->GetPitch()         * (parity ^ noTopBottom);
  const int dst_add_pitchUV = dst->GetPitch(plane1) * (parity ^ noTopBottom);
  const int dst_add_pitchA = dst->GetPitch(PLANAR_A) * (parity ^ noTopBottom);

  Copy(dst->GetWritePtr() + dst_add_pitch, dst->GetPitch() * 2,
    src->GetReadPtr() + src_add_pitch, src->GetPitch() * 2,
    src->GetRowSize(), src->GetHeight() >> 1, env);

  Copy(dst->GetWritePtr(plane1) + dst_add_pitchUV, dst->GetPitch(plane1) * 2,
    src->GetReadPtr(plane1) + src_add_pitchUV, src->GetPitch(plane1) * 2,
    src->GetRowSize(plane1), src->GetHeight(plane1) >> 1, env);

  Copy(dst->GetWritePtr(plane2) + dst_add_pitchUV, dst->GetPitch(plane2) * 2,
    src->GetReadPtr(plane2) + src_add_pitchUV, src->GetPitch(plane2) * 2,
    src->GetRowSize(plane2), src->GetHeight(plane2) >> 1, env);

  Copy(dst->GetWritePtr(PLANAR_A) + dst_add_pitchA, dst->GetPitch(PLANAR_A) * 2,
    src->GetReadPtr(PLANAR_A) + src_add_pitchA, src->GetPitch(PLANAR_A) * 2,
    src->GetRowSize(PLANAR_A), src->GetHeight(PLANAR_A) >> 1, env);
}

class DoubleWeaveFrames : public GenericVideoFilter
{
public:
  DoubleWeaveFrames(PClip _child)
    : GenericVideoFilter(_child)
  {
    vi.num_frames *= 2;
    if (vi.num_frames < 0)
      vi.num_frames = 0x7FFFFFFF; // MAXINT

    vi.MulDivFPS(2, 1);
  }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    if (!(n & 1))
    {
      return child->GetFrame(n >> 1, env);
    }
    else {
      PVideoFrame a = child->GetFrame(n >> 1, env);
      PVideoFrame b = child->GetFrame((n + 1) >> 1, env);
      bool parity = this->GetParity(n);

      if (a->IsWritable()) {
        copy_alternate_lines(a, b, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);
        return a;
      }
      else if (b->IsWritable()) {
        copy_alternate_lines(b, a, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
        return b;
      }
      else {
        PVideoFrame result = env->NewVideoFrame(vi);
        copy_alternate_lines(result, a, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), parity, env);
        copy_alternate_lines(result, b, vi.IsYUV() || vi.IsYUVA(), vi.IsPlanarRGB() || vi.IsPlanarRGBA(), !parity, env);
        return result;
      }
    }
  }

  inline bool __stdcall GetParity(int n) {
    return child->GetParity(n >> 1) ^ (n & 1);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    switch (cachehints) {
    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;
    case CACHE_GET_DEV_TYPE:
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }
};

/************************************
********   Factory Methods   *******
***********************************/

static AVSValue __cdecl Create_DoubleWeave(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsFieldBased())
    return new DoubleWeaveFields(clip);
  else
    return new DoubleWeaveFrames(clip);
}


static AVSValue __cdecl Create_Weave(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (!clip->GetVideoInfo().IsFieldBased())
    env->ThrowError("Weave: Weave should be applied on field-based material: use AssumeFieldBased() beforehand");
  AVSValue doubleWeave = Create_DoubleWeave(args, 0, env);
  return env->Invoke("SelectEven", AVSValue(&doubleWeave, 1));
}


/********************************
******  Invert filter  ******
********************************/


class Invert : public GenericVideoFilter
  /**
  * Class to invert selected RGBA channels
  **/
{
public:
  Invert(PClip _child, const char * _channels, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    switch (cachehints) {
    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;
    case CACHE_GET_DEV_TYPE:
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
  int mask;
  bool doB, doG, doR, doA;
  bool doY, doU, doV;

  unsigned __int64 mask64;
  int pixelsize;
  int bits_per_pixel; // 8,10..16
};

Invert::Invert(PClip _child, const char * _channels, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{
  doB = doG = doR = doA = doY = doU = doV = false;

  for (int k = 0; _channels[k] != '\0'; ++k) {
    switch (_channels[k]) {
    case 'B':
    case 'b':
      doB = true;
      break;
    case 'G':
    case 'g':
      doG = true;
      break;
    case 'R':
    case 'r':
      doR = true;
      break;
    case 'A':
    case 'a':
      doA = (vi.NumComponents() > 3);
      break;
    case 'Y':
    case 'y':
      doY = true;
      break;
    case 'U':
    case 'u':
      doU = (vi.NumComponents() > 1);
      break;
    case 'V':
    case 'v':
      doV = (vi.NumComponents() > 1);
      break;
    default:
      break;
    }
  }
  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  if (vi.IsYUY2()) {
    mask = doY ? 0x00ff00ff : 0;
    mask |= doU ? 0x0000ff00 : 0;
    mask |= doV ? 0xff000000 : 0;
  }
  else if (vi.IsRGB32()) {
    mask = doB ? 0x000000ff : 0;
    mask |= doG ? 0x0000ff00 : 0;
    mask |= doR ? 0x00ff0000 : 0;
    mask |= doA ? 0xff000000 : 0;
  }
  else if (vi.IsRGB64()) {
    mask64 = doB ? 0x000000000000ffffull : 0;
    mask64 |= (doG ? 0x00000000ffff0000ull : 0);
    mask64 |= (doR ? 0x0000ffff00000000ull : 0);
    mask64 |= (doA ? 0xffff000000000000ull : 0);
  }
  else {
    mask = 0xffffffff;
    mask64 = (1 << bits_per_pixel) - 1;
    mask64 |= (mask64 << 48) | (mask64 << 32) | (mask64 << 16); // works for 10 bit, too
                                                                // RGB24/48 is special case no use of this mask
  }
}

__device__ int dev_invert(int s, int mask0, int mask1) { return s ^ mask0; }
__device__ int2 dev_invert(int2 s, int mask0, int mask1) {
  int2 t = { s.x ^ mask0, s.y ^ mask1 };
  return t;
}
__device__ float4 dev_invert(float4 s, int mask0, int mask1) {
  return VHelper<float4>::make(1.0f) - s;
}

template <typename vpixel_t>
__global__ void kl_invert_plane(vpixel_t* ptr, int width, int height, int pitch, int mask0, int mask1)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    ptr[x + y * pitch] = dev_invert(ptr[x + y * pitch], mask0, mask1);
  }
}

template <typename vpixel_t>
static void launch_invert_plane(vpixel_t* ptr, int width, int height, int pitch, int mask0, int mask1, PNeoEnv env)
{
  dim3 threads(32, 8);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_invert_plane << <blocks, threads >> >(ptr, width, height, pitch, mask0, mask1);
  DEBUG_SYNC;
}

template <typename pixel_t>
__global__ void kl_invert_rgb(pixel_t* ptr, int width, int height, int el_pitch, int bMask, int gMask, int rMask)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    ptr[3 * x + 0 + y * el_pitch] = ptr[3 * x + 0 + y * el_pitch] ^ bMask;
    ptr[3 * x + 1 + y * el_pitch] = ptr[3 * x + 1 + y * el_pitch] ^ gMask;
    ptr[3 * x + 2 + y * el_pitch] = ptr[3 * x + 2 + y * el_pitch] ^ rMask;
  }
}

template <typename pixel_t>
static void launch_invert_rgb(pixel_t* ptr, int width, int height, int el_pitch, int bMask, int gMask, int rMask, PNeoEnv env)
{
  dim3 threads(32, 8);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_invert_rgb << <blocks, threads >> >(ptr, width, height, el_pitch, bMask, gMask, rMask);
  DEBUG_SYNC;
}

static void invert_frame_sse2(BYTE* frame, int pitch, int width, int height, int mask) {
  __m128i maskv = _mm_set1_epi32(mask);

  BYTE* endp = frame + pitch * height;

  while (frame < endp) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(frame));
    __m128i inv = _mm_xor_si128(src, maskv);
    _mm_store_si128(reinterpret_cast<__m128i*>(frame), inv);
    frame += 16;
  }
}

static void invert_frame_uint16_sse2(BYTE* frame, int pitch, int width, int height, uint64_t mask64) {
  __m128i maskv = _mm_set_epi32((uint32_t)(mask64 >> 32), (uint32_t)mask64, (uint32_t)(mask64 >> 32), (uint32_t)mask64);

  BYTE* endp = frame + pitch * height;

  while (frame < endp) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(frame));
    __m128i inv = _mm_xor_si128(src, maskv);
    _mm_store_si128(reinterpret_cast<__m128i*>(frame), inv);
    frame += 16;
  }
}

#ifdef X86_32

//mod4 width (in bytes) is required
static void invert_frame_mmx(BYTE* frame, int pitch, int width, int height, int mask)
{
  __m64 maskv = _mm_set1_pi32(mask);
  int mod8_width = width / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(frame + x);
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<__m64*>(frame + x) = inv;
    }

    if (mod8_width != width) {
      //last four pixels
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(frame + width - 4));
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<int*>(frame + width - 4) = _mm_cvtsi64_si32(inv);
    }
    frame += pitch;
  }
  _mm_empty();
}

static void invert_plane_mmx(BYTE* frame, int pitch, int width, int height)
{
#pragma warning(push)
#pragma warning(disable: 4309)
  __m64 maskv = _mm_set1_pi8(0xFF);
#pragma warning(pop)
  int mod8_width = width / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(frame + x);
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<__m64*>(frame + x) = inv;
    }

    for (int x = mod8_width; x < width; ++x) {
      frame[x] = frame[x] ^ 255;
    }
    frame += pitch;
  }
  _mm_empty();
}

#endif

//mod4 width is required
static void invert_frame_c(BYTE* frame, int pitch, int width, int height, int mask) {
  for (int y = 0; y < height; ++y) {
    int* intptr = reinterpret_cast<int*>(frame);

    for (int x = 0; x < width / 4; ++x) {
      intptr[x] = intptr[x] ^ mask;
    }
    frame += pitch;
  }
}

static void invert_frame_uint16_c(BYTE* frame, int pitch, int width, int height, uint64_t mask64) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 8; ++x) {
      reinterpret_cast<uint64_t *>(frame)[x] = reinterpret_cast<uint64_t *>(frame)[x] ^ mask64;
    }
    frame += pitch;
  }
}

static void invert_plane_c(BYTE* frame, int pitch, int row_size, int height) {
  int mod4_width = row_size / 4 * 4;
  for (int y = 0; y < height; ++y) {
    int* intptr = reinterpret_cast<int*>(frame);

    for (int x = 0; x < mod4_width / 4; ++x) {
      intptr[x] = intptr[x] ^ 0xFFFFFFFF;
    }

    for (int x = mod4_width; x < row_size; ++x) {
      frame[x] = frame[x] ^ 255;
    }
    frame += pitch;
  }
}

static void invert_plane_uint16_c(BYTE* frame, int pitch, int row_size, int height, uint64_t mask64) {
  int mod8_width = row_size / 8 * 8;
  uint16_t mask16 = mask64 & 0xFFFF; // for planes, all 16 bit parts of 64 bit mask is the same
  for (int y = 0; y < height; ++y) {

    for (int x = 0; x < mod8_width / 8; ++x) {
      reinterpret_cast<uint64_t *>(frame)[x] ^= mask64;
    }

    for (int x = mod8_width; x < row_size; ++x) {
      reinterpret_cast<uint16_t *>(frame)[x] ^= mask16;
    }
    frame += pitch;
  }
}

static void invert_plane_float_c(BYTE* frame, int pitch, int row_size, int height, bool chroma) {
  const int width = row_size / sizeof(float);
#ifdef FLOAT_CHROMA_IS_ZERO_CENTERED
  const float max = chroma ? 0.0f : 1.0f;
#else
  const float max = 1.0f;
#endif
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      reinterpret_cast<float *>(frame)[x] = max - reinterpret_cast<float *>(frame)[x];
    }
    frame += pitch;
  }
}

static void invert_frame(BYTE* frame, int pitch, int rowsize, int height, int mask, uint64_t mask64, int pixelsize, PNeoEnv env) {
  if (IS_CUDA) {
    if (pixelsize == 1) {
      launch_invert_plane((int*)frame, (rowsize + 3) >> 2, height, pitch >> 2, mask, 0, env);
    }
    else {
      launch_invert_plane((int2*)frame, (rowsize + 7) >> 3, height, pitch >> 3, (int)mask64, (int)(mask64 >> 32), env);
    }
  }
  else {
    if ((pixelsize == 1 || pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(frame, 16))
    {
      if (pixelsize == 1)
        invert_frame_sse2(frame, pitch, rowsize, height, mask);
      else
        invert_frame_uint16_sse2(frame, pitch, rowsize, height, mask64);
    }
#ifdef X86_32
    else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_MMX))
    {
      invert_frame_mmx(frame, pitch, rowsize, height, mask);
    }
#endif
    else
    {
      if (pixelsize == 1)
        invert_frame_c(frame, pitch, rowsize, height, mask);
      else
        invert_frame_uint16_c(frame, pitch, rowsize, height, mask64);
    }
  }
}

static void invert_frame_uint16(BYTE* frame, int pitch, int rowsize, int height, uint64_t mask64, IScriptEnvironment *env) {
  if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(frame, 16))
  {
    invert_frame_uint16_sse2(frame, pitch, rowsize, height, mask64);
  }
  else
  {
    invert_frame_uint16_c(frame, pitch, rowsize, height, mask64);
  }
}


static void invert_plane(BYTE* frame, int pitch, int rowsize, int height, int pixelsize, uint64_t mask64, bool chroma, PNeoEnv env) {
  if (IS_CUDA) {
    if (pixelsize == 1) {
      launch_invert_plane((int*)frame, (rowsize + 3) >> 2, height, pitch >> 2, 0xFFFFFFFF, 0, env);
    }
    else if (pixelsize == 2) {
      launch_invert_plane((int2*)frame, (rowsize + 7) >> 3, height, pitch >> 3, (int)mask64, (int)(mask64 >> 32), env);
    }
    else {
      launch_invert_plane((float4*)frame, (rowsize + 15) >> 4, height, pitch >> 4, 0, 0, env);
    }
  }
  else {
    if ((pixelsize == 1 || pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(frame, 16))
    {
      if (pixelsize == 1)
        invert_frame_sse2(frame, pitch, rowsize, height, 0xffffffff);
      else if (pixelsize == 2)
        invert_frame_uint16_sse2(frame, pitch, rowsize, height, mask64);
    }
#ifdef X86_32
    else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_MMX))
    {
      invert_plane_mmx(frame, pitch, rowsize, height);
    }
#endif
    else
    {
      if (pixelsize == 1)
        invert_plane_c(frame, pitch, rowsize, height);
      else if (pixelsize == 2)
        invert_plane_uint16_c(frame, pitch, rowsize, height, mask64);
      else {
        invert_plane_float_c(frame, pitch, rowsize, height, chroma);
      }
    }
  }
}

PVideoFrame Invert::GetFrame(int n, IScriptEnvironment* env_)
{
  PNeoEnv env = env_;

  PVideoFrame f = child->GetFrame(n, env);

  env->MakeWritable(&f);

  BYTE* pf = f->GetWritePtr();
  int pitch = f->GetPitch();
  int rowsize = f->GetRowSize();
  int height = f->GetHeight();

  if (vi.IsPlanar()) {
    // planar YUV
    if (vi.IsYUV() || vi.IsYUVA()) {
      if (doY)
        invert_plane(pf, pitch, f->GetRowSize(PLANAR_Y_ALIGNED), height, pixelsize, mask64, false, env);
      if (doU)
        invert_plane(f->GetWritePtr(PLANAR_U), f->GetPitch(PLANAR_U), f->GetRowSize(PLANAR_U_ALIGNED), f->GetHeight(PLANAR_U), pixelsize, mask64, true, env);
      if (doV)
        invert_plane(f->GetWritePtr(PLANAR_V), f->GetPitch(PLANAR_V), f->GetRowSize(PLANAR_V_ALIGNED), f->GetHeight(PLANAR_V), pixelsize, mask64, true, env);
    }
    // planar RGB
    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      if (doG) // first plane, GetWritePtr w/o parameters
        invert_plane(pf, pitch, f->GetRowSize(PLANAR_G_ALIGNED), height, pixelsize, mask64, false, env);
      if (doB)
        invert_plane(f->GetWritePtr(PLANAR_B), f->GetPitch(PLANAR_B), f->GetRowSize(PLANAR_B_ALIGNED), f->GetHeight(PLANAR_B), pixelsize, mask64, false, env);
      if (doR)
        invert_plane(f->GetWritePtr(PLANAR_R), f->GetPitch(PLANAR_R), f->GetRowSize(PLANAR_R_ALIGNED), f->GetHeight(PLANAR_R), pixelsize, mask64, false, env);
    }
    // alpha
    if (doA && (vi.IsPlanarRGBA() || vi.IsYUVA()))
      invert_plane(f->GetWritePtr(PLANAR_A), f->GetPitch(PLANAR_A), f->GetRowSize(PLANAR_A_ALIGNED), f->GetHeight(PLANAR_A), pixelsize, mask64, false, env);
  }
  else if (vi.IsYUY2() || vi.IsRGB32() || vi.IsRGB64()) {
    invert_frame(pf, pitch, rowsize, height, mask, mask64, pixelsize, env);
  }
  else if (vi.IsRGB24()) {
    int rMask = doR ? 0xff : 0;
    int gMask = doG ? 0xff : 0;
    int bMask = doB ? 0xff : 0;
    if (IS_CUDA) {
      launch_invert_rgb(pf, rowsize / 3, height, pitch, bMask, gMask, rMask, env);
    }
    else {
      for (int i = 0; i<height; i++) {

        for (int j = 0; j<rowsize; j += 3) {
          pf[j + 0] = pf[j + 0] ^ bMask;
          pf[j + 1] = pf[j + 1] ^ gMask;
          pf[j + 2] = pf[j + 2] ^ rMask;
        }
        pf += pitch;
      }
    }
  }
  else if (vi.IsRGB48()) {
    int rMask = doR ? 0xffff : 0;
    int gMask = doG ? 0xffff : 0;
    int bMask = doB ? 0xffff : 0;
    if (IS_CUDA) {
      launch_invert_rgb((uint16_t*)pf, rowsize / 6, height, pitch >> 1, bMask, gMask, rMask, env);
    }
    else {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < rowsize / pixelsize; j += 3) {
          reinterpret_cast<uint16_t *>(pf)[j + 0] ^= bMask;
          reinterpret_cast<uint16_t *>(pf)[j + 1] ^= gMask;
          reinterpret_cast<uint16_t *>(pf)[j + 2] ^= rMask;
        }
        pf += pitch;
      }
    }
  }

  return f;
}


AVSValue Invert::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Invert(args[0].AsClip(), args[0].AsClip()->GetVideoInfo().IsRGB() ? args[1].AsString("RGBA") : args[1].AsString("YUVA"), env);
}

extern const FuncDefinition generic_filters[] = {
  { "Align",  BUILTIN_FUNC_PREFIX,  "c", Align::Create, 0 },
  { "Weave",            BUILTIN_FUNC_PREFIX, "c", Create_Weave },
  { "DoubleWeave",      BUILTIN_FUNC_PREFIX, "c", Create_DoubleWeave },
  { "Invert",       BUILTIN_FUNC_PREFIX, "c[channels]s", Invert::Create },
  { 0 }
};
