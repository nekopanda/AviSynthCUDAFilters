
#include <stdint.h>
#include <avisynth.h>

#include <memory>
#include <algorithm>

#include "DeviceLocalData.h"
#include "CommonFunctions.h"
#include "VectorFunctions.cuh"
#include "KFMFilterBase.cuh"
#include "Copy.h"

#include "Frame.h"

#include "DeviceLocalData.cpp"

int GetDeviceTypes(const PClip& clip);

static int scaleParam(float thresh, int pixelBits)
{
  return (int)(thresh * (1 << (pixelBits - 8)) + 0.5f);
}

class KDebandBase : public GenericVideoFilter {
protected:
  int logUVx;
  int logUVy;

public:
  KDebandBase(PClip _child)
    : GenericVideoFilter(_child)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  { }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }
};

enum {
  MAX_TEMPORAL_DIST = 16
};

template<typename vpixel_t> struct TemporalNRPtrs {
  vpixel_t* out[1];
  const vpixel_t* in[MAX_TEMPORAL_DIST * 2 + 1];
};

template<typename vpixel_t>
__host__ __device__ void sum_pixel(int4 diff, int thresh, vpixel_t ref, int4& cnt, int4& sum)
{
  cnt.x += (diff.x <= thresh);
  cnt.y += (diff.y <= thresh);
  cnt.z += (diff.z <= thresh);
  cnt.w += (diff.w <= thresh);
  sum.x += ref.x * (diff.x <= thresh);
  sum.y += ref.y * (diff.y <= thresh);
  sum.z += ref.z * (diff.z <= thresh);
  sum.w += ref.w * (diff.w <= thresh);
}

template<typename vpixel_t>
__host__ __device__ void average_pixel(int4 sum, int4 cnt, vpixel_t& out)
{
#ifdef __CUDA_ARCH__
  // CUDA版は __fdividef を使う
//#if 0 // あまり変わらないのでCPU版と同じにする
  out.x = (int)(__fdividef(sum.x, cnt.x) + 0.5f);
  out.y = (int)(__fdividef(sum.y, cnt.y) + 0.5f);
  out.z = (int)(__fdividef(sum.z, cnt.z) + 0.5f);
  out.w = (int)(__fdividef(sum.w, cnt.w) + 0.5f);
#else
  out.x = (int)((float)sum.x / cnt.x + 0.5f);
  out.y = (int)((float)sum.y / cnt.y + 0.5f);
  out.z = (int)((float)sum.z / cnt.z + 0.5f);
  out.w = (int)((float)sum.w / cnt.w + 0.5f);
#endif
}

template<typename vpixel_t>
void cpu_temporal_nr(TemporalNRPtrs<vpixel_t>* data,
  int nframes, int mid, int width, int height, int pitch, int thresh)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      vpixel_t center = data->in[mid][x + y * pitch];

      int4 pixel_count = VHelper<int4>::make(0);
      int4 sum = VHelper<int4>::make(0);
      for (int i = 0; i < nframes; ++i) {
        vpixel_t ref = data->in[i][x + y * pitch];
        int4 diff = absdiff(ref, center);
        sum_pixel(diff, thresh, ref, pixel_count, sum);
      }

      average_pixel(sum, pixel_count, data->out[0][x + y * pitch]);
    }
  }
}

template<typename vpixel_t>
__global__ void kl_temporal_nr(
  const TemporalNRPtrs<vpixel_t>* __restrict__ data,
  int nframes, int mid, int width, int height, int pitch, int thresh)
{
  int tx = threadIdx.x;
  int x = tx + blockIdx.x * blockDim.x;
  //int b = threadIdx.y;
  int y = blockIdx.y;

  if (x < width) {
    vpixel_t center = data->in[mid][x + pitch * y];

    int4 pixel_count = VHelper<int4>::make(0);
    int4 sum = VHelper<int4>::make(0);
    for (int i = 0; i < nframes; ++i) {
      vpixel_t ref = data->in[i][x + pitch * y];
      int4 diff = absdiff(ref, center);
      sum_pixel(diff, thresh, ref, pixel_count, sum);
    }

    average_pixel(sum, pixel_count, data->out[0][x + y * pitch]);
  }
}

class KTemporalNR : public KDebandBase
{
  int dist;
  int thresh;

  template <typename pixel_t>
  PVideoFrame MakeFrames(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    int nframes = dist * 2 + 1;
    auto frames = std::unique_ptr<Frame[]>(new Frame[nframes]);
    for (int i = 0; i < nframes; ++i) {
      frames[i] = child->GetFrame(clamp(n - dist + i, 0, vi.num_frames - 1), env);
    }

    TemporalNRPtrs<vpixel_t> *ptrs = new TemporalNRPtrs<vpixel_t>[3];
    TemporalNRPtrs<vpixel_t>& ptrsY = ptrs[0];
    TemporalNRPtrs<vpixel_t>& ptrsU = ptrs[1];
    TemporalNRPtrs<vpixel_t>& ptrsV = ptrs[2];

    Frame dst = env->NewVideoFrame(vi);
    ptrsY.out[0] = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
    ptrsU.out[0] = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    ptrsV.out[0] = dst.GetWritePtr<vpixel_t>(PLANAR_V);
    for (int i = 0; i < nframes; ++i) {
      ptrsY.in[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_Y);
      ptrsU.in[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_U);
      ptrsV.in[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_V);
    }

    int pitchY = dst.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = dst.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;
    int mid = nframes / 2;

    if (IS_CUDA) {
      int work_bytes = sizeof(TemporalNRPtrs<vpixel_t>) * 3;
      VideoInfo workvi = VideoInfo();
      workvi.pixel_type = VideoInfo::CS_BGR32;
      workvi.width = 16;
      workvi.height = nblocks(work_bytes, workvi.width * 4);
      Frame work = env->NewVideoFrame(workvi);

      TemporalNRPtrs<vpixel_t>* dptrs =
        work.GetWritePtr<TemporalNRPtrs<vpixel_t>>();
      CUDA_CHECK(cudaMemcpyAsync(dptrs, ptrs, work_bytes, cudaMemcpyHostToDevice));

      dim3 threads(64);
      dim3 blocks(nblocks(width4, threads.x), vi.height);
      dim3 blocksUV(nblocks(width4UV, threads.x), heightUV);
      kl_temporal_nr << <blocks, threads, 0, stream >> > (
        &dptrs[0], nframes, mid, width4, vi.height, pitchY, thresh);
      DEBUG_SYNC;
      kl_temporal_nr << <blocksUV, threads, 0, stream >> > (
        &dptrs[1], nframes, mid, width4UV, heightUV, pitchUV, thresh);
      DEBUG_SYNC;
      kl_temporal_nr << <blocksUV, threads, 0, stream >> > (
        &dptrs[2], nframes, mid, width4UV, heightUV, pitchUV, thresh);
      DEBUG_SYNC;

      // 終わったら解放するコールバックを追加
      env->DeviceAddCallback([](void* arg) {
        delete[]((TemporalNRPtrs<vpixel_t>*)arg);
      }, ptrs);
    }
    else {
      cpu_temporal_nr(&ptrsY, nframes, mid, width4, vi.height, pitchY, thresh);
      cpu_temporal_nr(&ptrsU, nframes, mid, width4UV, heightUV, pitchUV, thresh);
      cpu_temporal_nr(&ptrsV, nframes, mid, width4UV, heightUV, pitchUV, thresh);
      delete[] ptrs;
    }

    return dst.frame;
  }

public:
  KTemporalNR(PClip clip, int dist, float thresh, IScriptEnvironment* env)
    : KDebandBase(clip)
    , dist(dist)
    , thresh(scaleParam(thresh, vi.BitsPerComponent()))
  {
    if (dist > MAX_TEMPORAL_DIST) {
      env->ThrowError("[KTemporalNR] maximum dist is 16");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return MakeFrames<uint8_t>(n, env);
    case 2:
      return MakeFrames<uint16_t>(n, env);
    default:
      env->ThrowError("[KTemporalNR] Unsupported pixel format");
      break;
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KDebandBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KTemporalNR(
      args[0].AsClip(),            // clip
      args[1].AsInt(3),            // dist
      (float)args[2].AsFloat(1),   // thresh
      env);
  }
};

// ランダムな128bit列をランダムな -range ～ range にして返す
// range は0～127以下
static __device__ __host__ int random_range(uint8_t random, char range) {
  return ((((range << 1) + 1) * (int)random) >> 8) - range;
}

template <typename pixel_t, int sample_mode, bool blur_first>
void cpu_reduce_banding(
  pixel_t* dst, const pixel_t* src, const uint8_t* rand,
  int width, int height, int pitch, int range, int thresh, PNeoEnv env)
{
  int rand_step = width * height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offset = y * pitch + x;

      int range_limited = min(min(range, y), min(height - y - 1, min(x, width - x - 1)));
      int refA = random_range(rand[offset + rand_step * 0], range_limited);
      int refB = random_range(rand[offset + rand_step * 1], range_limited);

      int src_val = src[offset];
      int avg, diff;

      if (sample_mode == 0) {
        int ref = refA * pitch + refB;

        avg = src[offset + ref];
        diff = absdiff(src_val, avg);

      }
      else if (sample_mode == 1) {
        int ref = refA * pitch + refB;

        int ref_p = src[offset + ref];
        int ref_m = src[offset - ref];

        avg = (ref_p + ref_m) >> 1;
        diff = blur_first
          ? absdiff(src_val, avg)
          : max(absdiff(src_val, ref_p),
            absdiff(src_val, ref_m));
      }
      else {
        int ref_0 = refA * pitch + refB;
        int ref_1 = refA - refB * pitch;

        int ref_0p = src[offset + ref_0];
        int ref_0m = src[offset - ref_0];
        int ref_1p = src[offset + ref_1];
        int ref_1m = src[offset - ref_1];

        avg = (ref_0p + ref_0m + ref_1p + ref_1m) >> 2;
        diff = blur_first
          ? absdiff(src_val, avg)
          : max(
            max(absdiff(src_val, ref_0p), absdiff(src_val, ref_0m)),
            max(absdiff(src_val, ref_1p), absdiff(src_val, ref_1m)));
      }

      dst[offset] = (diff <= thresh) ? avg : src_val;
    }
  }
}

template <typename pixel_t, int sample_mode, bool blur_first>
__global__ void kl_reduce_banding(
  pixel_t* __restrict__ dst, const pixel_t* __restrict__ src, const uint8_t* __restrict__ rand,
  int width, int height, int pitch, int range, int thresh)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int rand_step = width * height;
  const int offset = y * pitch + x;

  if (x < width && y < height) {

    int range_limited = min(min(range, y), min(height - y - 1, min(x, width - x - 1)));
    int refA = random_range(rand[offset + rand_step * 0], range_limited);
    int refB = random_range(rand[offset + rand_step * 1], range_limited);

    int src_val = src[offset];
    int avg, diff;

    if (sample_mode == 0) {
      int ref = refA * pitch + refB;

      avg = src[offset + ref];
      diff = absdiff(src_val, avg);

    }
    else if (sample_mode == 1) {
      int ref = refA * pitch + refB;

      int ref_p = src[offset + ref];
      int ref_m = src[offset - ref];

      avg = (ref_p + ref_m) >> 1;
      diff = blur_first
        ? absdiff(src_val, avg)
        : max(absdiff(src_val, ref_p),
          absdiff(src_val, ref_m));
    }
    else {
      int ref_0 = refA * pitch + refB;
      int ref_1 = refA - refB * pitch;

      int ref_0p = src[offset + ref_0];
      int ref_0m = src[offset - ref_0];
      int ref_1p = src[offset + ref_1];
      int ref_1m = src[offset - ref_1];

      avg = (ref_0p + ref_0m + ref_1p + ref_1m) >> 2;
      diff = blur_first
        ? absdiff(src_val, avg)
        : max(
          max(absdiff(src_val, ref_0p), absdiff(src_val, ref_0m)),
          max(absdiff(src_val, ref_1p), absdiff(src_val, ref_1m)));
    }

    dst[offset] = (diff <= thresh) ? avg : src_val;
  }
}

template <typename pixel_t, int sample_mode, bool blur_first>
void launch_reduce_banding(
  pixel_t* dst, const pixel_t* src, const uint8_t* rand,
  int width, int height, int pitch, int range, int thresh, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_reduce_banding<pixel_t, sample_mode, blur_first> << <blocks, threads, 0, stream >> > (
    dst, src, rand, width, height, pitch, range, thresh);
}

struct XorShift {
  uint32_t x, y, z, w;
  XorShift(int seed) {
    x = 123456789 - seed;
    y = 362436069;
    z = 521288629;
    w = 88675123;
  }
  uint32_t next() {
    uint32_t t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    w ^= t ^ (t >> 8) ^ (w >> 19);
    return w;
  }
};

class KDeband : public KDebandBase
{
  int range;
  int thresh;
  int sample_mode;
  bool blur_first;

  std::unique_ptr<DeviceLocalData<uint8_t>> rand;

  DeviceLocalData<uint8_t>* CreateDebandRandom(int width, int height, int seed, PNeoEnv env)
  {
    const int max_per_pixel = 2;
    int length = width * height * max_per_pixel;
    auto rand_buf = std::unique_ptr<uint8_t[]>(new uint8_t[length]);

    XorShift xor (seed);

    int i = 0;
    for (; i <= length - 4; i += 4) {
      *(uint32_t*)(&rand_buf[i]) = xor.next();
    }
    if (i < length) {
      auto r = xor.next();
      memcpy(&rand_buf[i], &r, length - i);
    }

    return new DeviceLocalData<uint8_t>(rand_buf.get(), length, env);
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame src = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

    const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

    int pitchY = src.GetPitch<pixel_t>(PLANAR_Y);
    int pitchUV = src.GetPitch<pixel_t>(PLANAR_U);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    const uint8_t* prand = rand->GetData(env);

    void(*table[2][6])(
      pixel_t* dst, const pixel_t* src, const uint8_t* rand,
      int width, int height, int pitch, int range, int thresh, PNeoEnv env) =
    {
      {
        cpu_reduce_banding<pixel_t, 0, false>,
        cpu_reduce_banding<pixel_t, 0, true>,
        cpu_reduce_banding<pixel_t, 1, false>,
        cpu_reduce_banding<pixel_t, 1, true>,
        cpu_reduce_banding<pixel_t, 2, false>,
        cpu_reduce_banding<pixel_t, 2, true>,
      },
      {
        launch_reduce_banding<pixel_t, 0, false>,
        launch_reduce_banding<pixel_t, 0, true>,
        launch_reduce_banding<pixel_t, 1, false>,
        launch_reduce_banding<pixel_t, 1, true>,
        launch_reduce_banding<pixel_t, 2, false>,
        launch_reduce_banding<pixel_t, 2, true>,
      }
    };

    int table_idx = sample_mode * 2 + (blur_first ? 1 : 0);

    if (IS_CUDA) {
      table[1][table_idx](dstY, srcY, prand, vi.width, vi.height, pitchY, range, thresh, env);
      DEBUG_SYNC;
      table[1][table_idx](dstU, srcU, prand, widthUV, heightUV, pitchUV, range, thresh, env);
      DEBUG_SYNC;
      table[1][table_idx](dstV, srcV, prand, widthUV, heightUV, pitchUV, range, thresh, env);
      DEBUG_SYNC;
    }
    else {
      table[0][table_idx](dstY, srcY, prand, vi.width, vi.height, pitchY, range, thresh, env);
      table[0][table_idx](dstU, srcU, prand, widthUV, heightUV, pitchUV, range, thresh, env);
      table[0][table_idx](dstV, srcV, prand, widthUV, heightUV, pitchUV, range, thresh, env);
    }

    return dst.frame;
  }

public:
  KDeband(PClip clip, int range, float thresh, int sample_mode, bool blur_first, PNeoEnv env)
    : KDebandBase(clip)
    , range(range)
    , thresh(scaleParam(thresh, vi.BitsPerComponent()))
    , sample_mode(sample_mode)
    , blur_first(blur_first)
    , rand(CreateDebandRandom(vi.width, vi.height, 0, env))
  {
    if (sample_mode != 0 && sample_mode != 1 && sample_mode != 2) {
      env->ThrowError("[KDeband] sample_mode must be 0,1,2");
    }
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KDebandBase::SetCacheHints(cachehints, frame_range);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KDeband] Unsupported pixel format");
      break;
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KDeband(
      args[0].AsClip(),          // clip
      args[1].AsInt(25),         // range
      (float)args[2].AsFloat(1), // thresh
      args[3].AsInt(1),          // sample_mode
      args[4].AsBool(false),     // blur_first
      env);
  }
};

template <typename pixel_t>
void cpu_copy(pixel_t* dst, const pixel_t* __restrict__ src, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_copy(pixel_t* dst, const pixel_t* __restrict__ src, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * pitch] = src[x + y * pitch];
  }
}

template <typename pixel_t>
void cpu_fill(pixel_t* dst, pixel_t v, int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * pitch] = v;
    }
  }
}

template <typename pixel_t>
__global__ void kl_fill(pixel_t* dst, pixel_t v, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * pitch] = v;
  }
}

enum {
  EDGE_CHECK_NONE = 16,
  EDGE_CHECK_DARK = 180,
  EDGE_CHECK_BRIGHT = 120,
  EDGE_CHECK_BLACK = 240,
  EDGE_CHECK_WHITE = 50,
};
#define SCALE(c) (int)(((float)c / 255.0f) * maxv)

template <typename pixel_t, bool selective>
__device__ __host__ void dev_el_min_max(int& hmin, int& hmax, const pixel_t* __restrict__ src, int pitch)
{
	enum { S = (int)selective };
	int vmax, vmin;
	hmax = hmin = src[-(2 + S)];
	vmax = vmin = src[-(2 + S) * pitch];
	for (int i = -(1 + S); i < (3 + S); ++i) {
		int hcur = src[i];
		int vcur = src[i*pitch];
		hmax = max(hmax, hcur);
		hmin = min(hmin, hcur);
		vmax = max(vmax, vcur);
		vmin = min(vmin, vcur);
	}
	if (hmax - hmin < vmax - vmin) {
		hmax = vmax, hmin = vmin;
	}
}

template <typename pixel_t, bool check, bool selective, bool uv>
void cpu_edgelevel(
	pixel_t* dstY_, pixel_t* dstU_, pixel_t* dstV_,
	const pixel_t* srcY_, const pixel_t* srcU_, const pixel_t* srcV_,
  int width, int height, int pitch, int maxv, int str, int strUV, int thrs, PNeoEnv env)
{
  enum { S = (int)selective };
  for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = y * pitch + x;
			const pixel_t* srcY = srcY_ + offset;
			const pixel_t* srcU = srcU_ + offset;
			const pixel_t* srcV = srcV_ + offset;
			pixel_t* dstY = dstY_ + offset;
			pixel_t* dstU = dstU_ + offset;
			pixel_t* dstV = dstV_ + offset;

			if (y <= (1 + S) || y >= height - (2 + S) || x <= (1 + S) || x >= width - (2 + S)) {
				if (x < width && y < height) {
					*dstY = check ? SCALE(EDGE_CHECK_NONE) : *srcY;
					if (uv) {
						*dstU = *srcU;
						*dstV = *srcV;
					}
				}
			}
			else {
				int hmax, hmin, vmax, vmin;
				int hdiffmax = 0, vdiffmax = 0;
				int hprev = hmax = hmin = srcY[-(2 + S)];
				int vprev = vmax = vmin = srcY[-(2 + S) * pitch];

				for (int i = -(1 + S); i < (3 + S); ++i) {
					int hcur = srcY[i];
					int vcur = srcY[i*pitch];

					hmax = max(hmax, hcur);
					hmin = min(hmin, hcur);
					vmax = max(vmax, vcur);
					vmin = min(vmin, vcur);

					if (selective) {
						hdiffmax = max(hdiffmax, abs(hcur - hprev));
						vdiffmax = max(vdiffmax, abs(vcur - vprev));
					}

					hprev = hcur;
					vprev = vcur;
				}

				if (hmax - hmin < vmax - vmin) {
					hmax = vmax, hmin = vmin;
				}
				hdiffmax = max(hdiffmax, vdiffmax);

				float rdiff = hdiffmax / (float)(hmax - hmin);

				// ～0.25: エッジでない可能性が高いのでなし
				// 0.25～0.35: ボーダー
				// 0.35～0.45: 甘いエッジなので強化
				// 0.45～0.55: ボーダー
				// 0.55～: 十分エッジなのでなし
				float factor = selective ? (clamp((0.55f - rdiff) * 10.0f, 0.0f, 1.0f) - clamp((0.35f - rdiff) * 10.0f, 0.0f, 1.0f)) : 1.0f;

				int minU, maxU;
				int minV, maxV;
				if (uv) {
					dev_el_min_max<pixel_t, selective>(minU, maxU, srcU, pitch);
					dev_el_min_max<pixel_t, selective>(minV, maxV, srcV, pitch);
				}

				int srcvY, srcvU, srcvV;
				srcvY = *srcY;
				if (uv) {
					srcvU = *srcU;
					srcvV = *srcV;
				}

				int dstvY, dstvU, dstvV;
				if (check) {
					if (hmax - hmin > thrs && factor > 0.0f) {
						int avgY = (hmax + hmin) >> 1;
						if (srcvY > avgY) {
							dstvY = (factor == 1.0f) ? SCALE(EDGE_CHECK_WHITE) : SCALE(EDGE_CHECK_BRIGHT);
						}
						else {
							dstvY = (factor == 1.0f) ? SCALE(EDGE_CHECK_BLACK) : SCALE(EDGE_CHECK_DARK);
						}
					}
					else {
						dstvY = SCALE(EDGE_CHECK_NONE);
					}
					if (uv) {
						dstvU = srcvU;
						dstvV = srcvV;
					}
				}
				else {
					if (hmax - hmin > thrs && factor > 0.0f) {
						float factorY = (str * factor) * 0.0625f;
						int avgY = (hmin + hmax) >> 1;
						dstvY = clamp(srcvY + (int)((srcvY - avgY) * factorY), hmin, hmax);
						dstvY = clamp(dstvY, 0, maxv);

						if (uv) {
							float factorUV = strUV * 0.0625f;
							int avgU = (minU + maxU) >> 1;
							dstvU = clamp(srcvU + (int)((srcvU - avgU) * factorUV), minU, maxU);
							dstvU = clamp(dstvU, 0, maxv);
							int avgV = (minV + maxV) >> 1;
							dstvV = clamp(srcvV + (int)((srcvV - avgV) * factorUV), minV, maxV);
							dstvV = clamp(dstvV, 0, maxv);
						}
					}
					else {
						dstvY = srcvY;
						if (uv) {
							dstvU = srcvU;
							dstvV = srcvV;
						}
					}
				}

				*dstY = dstvY;
				if (uv) {
					*dstU = dstvU;
					*dstV = dstvV;
				}
			}
		}
  }
}

template <typename pixel_t, bool check, bool selective, bool uv>
__global__ void kl_edgelevel(
	pixel_t* __restrict__ dstY,
	pixel_t* __restrict__ dstU,
	pixel_t* __restrict__ dstV,
	const pixel_t* __restrict__ srcY,
	const pixel_t* __restrict__ srcU,
	const pixel_t* __restrict__ srcV,
  int width, int height, int pitch, int maxv, int str, int strUV, int thrs)
{
  enum { S = (int)selective };
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = y * pitch + x;
  srcY += offset;
	if (uv) {
		srcU += offset;
		srcV += offset;
	}

	if (y <= (1 + S) || y >= height - (2 + S) || x <= (1 + S) || x >= width - (2 + S)) {
		if (x < width && y < height) {
			dstY[offset] = check ? SCALE(EDGE_CHECK_NONE) : *srcY;
			if (uv) {
				dstU[offset] = *srcU;
				dstV[offset] = *srcV;
			}
		}
	}
	else {
		int hmax, hmin, vmax, vmin;
		int hdiffmax = 0, vdiffmax = 0;
		int hprev = hmax = hmin = srcY[-(2 + S)];
		int vprev = vmax = vmin = srcY[-(2 + S) * pitch];

		for (int i = -(1 + S); i < (3 + S); ++i) {
			int hcur = srcY[i];
			int vcur = srcY[i*pitch];

			hmax = max(hmax, hcur);
			hmin = min(hmin, hcur);
			vmax = max(vmax, vcur);
			vmin = min(vmin, vcur);

			if (selective) {
				hdiffmax = max(hdiffmax, abs(hcur - hprev));
				vdiffmax = max(vdiffmax, abs(vcur - vprev));
			}

			hprev = hcur;
			vprev = vcur;
		}

		if (hmax - hmin < vmax - vmin) {
			hmax = vmax, hmin = vmin;
		}
		hdiffmax = max(hdiffmax, vdiffmax);

		float rdiff = hdiffmax / (float)(hmax - hmin);

		// ～0.25: エッジでない可能性が高いのでなし
		// 0.25～0.35: ボーダー
		// 0.35～0.45: 甘いエッジなので強化
		// 0.45～0.55: ボーダー
		// 0.55～: 十分エッジなのでなし
		float factor = selective ? (clamp((0.55f - rdiff) * 10.0f, 0.0f, 1.0f) - clamp((0.35f - rdiff) * 10.0f, 0.0f, 1.0f)) : 1.0f;

		int srcvY = *srcY;
		int dstvY, dstvU, dstvV;
		if (check) {
			if (hmax - hmin > thrs && factor > 0.0f) {
				int avgY = (hmax + hmin) >> 1;
				if (srcvY > avgY) {
					dstvY = (factor == 1.0f) ? SCALE(EDGE_CHECK_WHITE) : SCALE(EDGE_CHECK_BRIGHT);
				}
				else {
					dstvY = (factor == 1.0f) ? SCALE(EDGE_CHECK_BLACK) : SCALE(EDGE_CHECK_DARK);
				}
			}
			else {
				dstvY = SCALE(EDGE_CHECK_NONE);
			}
			if (uv) {
				dstvU = *srcU;
				dstvV = *srcV;
			}
		}
		else {
			if (hmax - hmin > thrs && factor > 0.0f) {
				float factorY = (str * factor) * 0.0625f;
				int avgY = (hmin + hmax) >> 1;
				dstvY = clamp(srcvY + (int)((srcvY - avgY) * factorY), hmin, hmax);
				dstvY = clamp(dstvY, 0, maxv);

				if (uv) {
					float factorUV = strUV * 0.0625f;

					int minU, maxU;
					dev_el_min_max<pixel_t, selective>(minU, maxU, srcU, pitch);
					int avgU = (minU + maxU) >> 1;
					int srcvU = *srcU;
					dstvU = clamp(srcvU + (int)((srcvU - avgU) * factorUV), minU, maxU);
					dstvU = clamp(dstvU, 0, maxv);

					int minV, maxV;
					dev_el_min_max<pixel_t, selective>(minV, maxV, srcV, pitch);
					int avgV = (minV + maxV) >> 1;
					int srcvV = *srcV;
					dstvV = clamp(srcvV + (int)((srcvV - avgV) * factorUV), minV, maxV);
					dstvV = clamp(dstvV, 0, maxv);
				}
			}
			else {
				dstvY = srcvY;
				if (uv) {
					dstvU = *srcU;
					dstvV = *srcV;
				}
			}
		}

		dstY[offset] = dstvY;
		if (uv) {
			dstU[offset] = dstvU;
			dstV[offset] = dstvV;
		}
	}
}

template <typename pixel_t, bool check, bool selective, bool uv>
void launch_edgelevel(
	pixel_t* dstY, pixel_t* dstU, pixel_t* dstV,
	const pixel_t* srcY, const pixel_t* srcU, const pixel_t* srcV,
  int width, int height, int pitch, int maxv, int str, int strUV, int thrs, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_edgelevel<pixel_t, check, selective, uv> << <blocks, threads, 0, stream >> > (
		dstY, dstU, dstV, srcY, srcU, srcV, width, height, pitch, maxv, str, strUV, thrs);
}

template<typename T, typename CompareAndSwap>
__device__ __host__ void dev_sort_8elem(T& a0, T& a1, T& a2, T& a3, T& a4, T& a5, T& a6, T& a7)
{
  CompareAndSwap cas;

  // Batcher's odd-even mergesort
  // 8要素なら19comparisonなので最小のソーティングネットワークになるっぽい
  cas(a0, a1);
  cas(a2, a3);
  cas(a4, a5);
  cas(a6, a7);

  cas(a0, a2);
  cas(a1, a3);
  cas(a4, a6);
  cas(a5, a7);

  cas(a1, a2);
  cas(a5, a6);

  cas(a0, a4);
  cas(a1, a5);
  cas(a2, a6);
  cas(a3, a7);

  cas(a2, a4);
  cas(a3, a5);

  cas(a1, a2);
  cas(a3, a4);
  cas(a5, a6);
}

struct IntCompareAndSwap {
  __device__ __host__ void operator()(int& a, int& b) {
    int a_ = min(a, b);
    int b_ = max(a, b);
    a = a_; b = b_;
  }
};

template <typename pixel_t, int N>
void cpu_edgelevel_repair(
  pixel_t* dst, const pixel_t* el, const pixel_t* src,
  int width, int height, int pitch, PNeoEnv env)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // elはedgelevelの出力であること前提でボーダー処理は入れない
      // 必ずボーダー付近はel==srcであること
      // そうでないとメモリエラーが発生する

      int srcv = src[x + y * pitch];
      int elv = el[x + y * pitch];
      int dstv = srcv;

      if (elv != srcv) {
        int a0 = src[x - 1 + (y - 1) * pitch];
        int a1 = src[x + (y - 1) * pitch];
        int a2 = src[x + 1 + (y - 1) * pitch];
        int a3 = src[x - 1 + y * pitch];
        int a4 = src[x + 1 + y * pitch];
        int a5 = src[x - 1 + (y + 1) * pitch];
        int a6 = src[x + (y + 1) * pitch];
        int a7 = src[x + 1 + (y + 1) * pitch];

        dev_sort_8elem<int, IntCompareAndSwap>(a0, a1, a2, a3, a4, a5, a6, a7);

        switch (N) {
        case 1: // 1st
          dstv = clamp(elv, min(srcv, a0), max(srcv, a7));
          break;
        case 2: // 2nd
          dstv = clamp(elv, min(srcv, a1), max(srcv, a6));
          break;
        case 3: // 3rd
          dstv = clamp(elv, min(srcv, a2), max(srcv, a5));
          break;
        case 4: // 4th
          dstv = clamp(elv, min(srcv, a3), max(srcv, a4));
          break;
        }
      }

      dst[x + y * pitch] = dstv;
    }
  }
}

template <typename pixel_t, int N>
__global__ void kl_edgelevel_repair(
  pixel_t* __restrict__ dst, const pixel_t* __restrict__ el, const pixel_t* __restrict__ src,
  int width, int height, int pitch)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // elはedgelevelの出力であること前提でボーダー処理は入れない
    // 必ずボーダー付近はel==srcであること
    // そうでないとメモリエラーが発生する

    int srcv = src[x + y * pitch];
    int elv = el[x + y * pitch];
    int dstv = srcv;

    if (elv != srcv) {
      int a0 = src[x - 1 + (y - 1) * pitch];
      int a1 = src[x + (y - 1) * pitch];
      int a2 = src[x + 1 + (y - 1) * pitch];
      int a3 = src[x - 1 + y * pitch];
      int a4 = src[x + 1 + y * pitch];
      int a5 = src[x - 1 + (y + 1) * pitch];
      int a6 = src[x + (y + 1) * pitch];
      int a7 = src[x + 1 + (y + 1) * pitch];

      dev_sort_8elem<int, IntCompareAndSwap>(a0, a1, a2, a3, a4, a5, a6, a7);

      switch (N) {
      case 1: // 1st
        dstv = clamp(elv, min(srcv, a0), max(srcv, a7));
        break;
      case 2: // 2nd
        dstv = clamp(elv, min(srcv, a1), max(srcv, a6));
        break;
      case 3: // 3rd
        dstv = clamp(elv, min(srcv, a2), max(srcv, a5));
        break;
      case 4: // 4th
        dstv = clamp(elv, min(srcv, a3), max(srcv, a4));
        break;
      }
    }

    dst[x + y * pitch] = dstv;
  }
}

template <typename pixel_t, int N>
void launch_edgelevel_repair(
  pixel_t* dsttop, const pixel_t* __restrict__ eltop, const pixel_t* srctop,
  int width, int height, int pitch, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_edgelevel_repair<pixel_t, N> << <blocks, threads, 0, stream >> > (
    dsttop, eltop, srctop, width, height, pitch);
}

// スレッド数=(width,height)=srcのサイズ
template <typename pixel_t, int logUVx, int logUVy>
void cpu_el_to444(
	pixel_t* dst, const pixel_t* src,
	int width, int height, int dstPitch, int srcPitch, PNeoEnv env)
{
	enum {
		BW = (1 << logUVx),
		BH = (1 << logUVy),
	};

	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int v00 = src[(x + 0) + (y + 0) * srcPitch];
			dst[BW * x + BH * y * dstPitch] = v00;
			if (logUVx) {
				int v10 = src[(x + 1) + (y + 0) * srcPitch];
				dst[BW * x + 1 + BH * y * dstPitch] = (v00 + v10 + 1) >> 1;
				if (logUVy) {
					int v01 = src[(x + 0) + (y + 1) * srcPitch];
					int v11 = src[(x + 1) + (y + 1) * srcPitch];
					dst[BW * x + 0 + (BH * y + 1) * dstPitch] = (v00 + v01 + 1) >> 1;
					dst[BW * x + 1 + (BH * y + 1) * dstPitch] = (v00 + v10 + v01 + v11 + 2) >> 2;
				}
			}
		}
	}

	for (int y = 0; y < height - 1; ++y) {
		int x = width - 1;
		int v00 = src[(x + 0) + (y + 0) * srcPitch];
		dst[BW * x + BH * y * dstPitch] = v00;
		if (logUVx) {
			dst[BW * x + 1 + BH * y * dstPitch] = v00;
			if (logUVy) {
				int v01 = src[(x + 0) + (y + 1) * srcPitch];
				dst[BW * x + 0 + (BH * y + 1) * dstPitch] = (v00 + v01 + 1) >> 1;
				dst[BW * x + 1 + (BH * y + 1) * dstPitch] = (v00 + v01 + 1) >> 1;
			}
		}
	}

	for (int x = 0; x < width - 1; ++x) {
		int y = height - 1;
		int v00 = src[(x + 0) + (y + 0) * srcPitch];
		dst[BW * x + BH * y * dstPitch] = v00;
		if (logUVx) {
			int v10 = src[(x + 1) + (y + 0) * srcPitch];
			dst[BW * x + 1 + BH * y * dstPitch] = (v00 + v10 + 1) >> 1;
			if (logUVy) {
				dst[BW * x + 0 + (BH * y + 1) * dstPitch] = v00;
				dst[BW * x + 1 + (BH * y + 1) * dstPitch] = (v00 + v10 + 1) >> 1;
			}
		}
	}

	int x = width - 1;
	int y = height - 1;
	int v00 = src[(x + 0) + (y + 0) * srcPitch];
	dst[BW * x + BH * y * dstPitch] = v00;
	if (logUVx) {
		dst[BW * x + 1 + BH * y * dstPitch] = v00;
		if (logUVy) {
			dst[BW * x + 0 + (BH * y + 1) * dstPitch] = v00;
			dst[BW * x + 1 + (BH * y + 1) * dstPitch] = v00;
		}
	}
}

template <typename pixel_t, int logUVx, int logUVy>
__global__ void kl_el_to444(
	pixel_t* dst, const pixel_t* __restrict__ src,
	int width, int height, int dstPitch, int srcPitch)
{
	enum {
		BW = (1 << logUVx),
		BH = (1 << logUVy),
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int v00 = src[(x + 0) + (y + 0) * srcPitch];
		dst[BW * x + BH * y * dstPitch] = v00;
		if (logUVx) {
			int v10 = (x + 1 < width) ? src[(x + 1) + (y + 0) * srcPitch] : v00;
			dst[BW * x + 1 + BH * y * dstPitch] = (v00 + v10 + 1) >> 1;
			if (logUVy) {
				int v01, v11;
				if (y + 1 < height) {
					v01 = src[(x + 0) + (y + 1) * srcPitch];
					v11 = (x + 1 < width) ? src[(x + 1) + (y + 1) * srcPitch] : v01;
				}
				else {
					v01 = v00;
					v11 = (x + 1 < width) ? v10 : v00;
				}
				dst[BW * x + 0 + (BH * y + 1) * dstPitch] = (v00 + v01 + 1) >> 1;
				dst[BW * x + 1 + (BH * y + 1) * dstPitch] = (v00 + v10 + v01 + v11 + 2) >> 2;
			}
		}
	}
}

template <typename pixel_t, int logUVx, int logUVy>
void launch_el_to444(
	pixel_t* dst, const pixel_t* src,
	int width, int height, int dstPitch, int srcPitch, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
	dim3 threads(32, 16);
	dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
	kl_el_to444<pixel_t, logUVx, logUVy> << <blocks, threads, 0, stream >> > (
		dst, src, width, height, dstPitch, srcPitch);
}

// スレッド数=(width,height)=dstのサイズ
template <typename pixel_t, int logUVx, int logUVy>
void cpu_el_from444(
	pixel_t* dst, const pixel_t* src,
	int width, int height, int dstPitch, int srcPitch, PNeoEnv env)
{
	enum {
		BW = (1 << logUVx),
		BH = (1 << logUVy),
	};

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[x + y * dstPitch] = src[BW * x + BH * y * srcPitch];
		}
	}
}

template <typename pixel_t, int logUVx, int logUVy>
__global__ void kl_el_from444(
	pixel_t* dst, const pixel_t* __restrict__ src,
	int width, int height, int dstPitch, int srcPitch)
{
	enum {
		BW = (1 << logUVx),
		BH = (1 << logUVy),
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		dst[x + y * dstPitch] = src[BW * x + BH * y * srcPitch];
	}
}

template <typename pixel_t, int logUVx, int logUVy>
void launch_el_from444(
	pixel_t* dst, const pixel_t* src,
	int width, int height, int dstPitch, int srcPitch, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
	dim3 threads(32, 16);
	dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
	kl_el_from444<pixel_t, logUVx, logUVy> << <blocks, threads, 0, stream >> > (
		dst, src, width, height, dstPitch, srcPitch);
}

class KEdgeLevel : public KDebandBase
{
	int str;
	int strUV;
  int thrs;
  int repair;
  bool uv;
  bool show;

  int logUVx;
  int logUVy;

  template <typename pixel_t>
  void CopyUV(Frame& dst, Frame& src, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    const vpixel_t* srcU = src.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* srcV = src.GetReadPtr<vpixel_t>(PLANAR_V);
    vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchUV = src.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_copy << <blocksUV, threads, 0, stream >> > (dstU, srcU, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_copy << <blocksUV, threads, 0, stream >> > (dstV, srcV, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_copy<vpixel_t>(dstU, srcU, width4UV, heightUV, pitchUV);
      cpu_copy<vpixel_t>(dstV, srcV, width4UV, heightUV, pitchUV);
    }
  }

  template <typename pixel_t>
  void ClearUV(Frame& dst, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchUV = dst.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    vpixel_t zerov = VHelper<vpixel_t>::make(1 << (vi.BitsPerComponent() - 1));

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_fill << <blocksUV, threads, 0, stream >> > (dstU, zerov, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_fill << <blocksUV, threads, 0, stream >> > (dstV, zerov, width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_fill<vpixel_t>(dstU, zerov, width4UV, heightUV, pitchUV);
      cpu_fill<vpixel_t>(dstV, zerov, width4UV, heightUV, pitchUV);
    }
  }

	template <typename pixel_t>
	PVideoFrame GetUVFrame(Frame& src, PNeoEnv env) {
		if (logUVx == 0) {
			return src.frame;
		}
		VideoInfo vi444 = vi;
		vi444.pixel_type = Get444Type(vi);
		Frame dst = env->NewVideoFrame(vi444);
		const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
		const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
		pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
		pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);
		int srcPitch = src.GetPitch<pixel_t>(PLANAR_U);
		int dstPitch = dst.GetPitch<pixel_t>(PLANAR_U);
		void(*table[2][8])(
			pixel_t* dst, const pixel_t* src,
			int width, int height, int dstPitch, int srcPitch, PNeoEnv env) =
		{
			{ // CPU
				cpu_el_to444<pixel_t, 1, 0>,
				cpu_el_to444<pixel_t, 1, 1>,
			},
			{ // CUDA
				launch_el_to444<pixel_t, 1, 0>,
				launch_el_to444<pixel_t, 1, 1>,
			}
		};
		if (IS_CUDA) {
			table[1][logUVy](dstU, srcU, vi.width, vi.height, dstPitch, srcPitch, env);
			DEBUG_SYNC;
			table[1][logUVy](dstV, srcV, vi.width, vi.height, dstPitch, srcPitch, env);
			DEBUG_SYNC;
		}
		else {
			table[0][logUVy](dstU, srcU, vi.width, vi.height, dstPitch, srcPitch, env);
			table[0][logUVy](dstV, srcV, vi.width, vi.height, dstPitch, srcPitch, env);
		}
		return dst.frame;
	}

	template <typename pixel_t>
	PVideoFrame GetOutFrame(Frame& src, PNeoEnv env) {
		if (logUVx == 0) {
			return src.frame;
		}
		Frame dst = env->NewVideoFrame(vi);
		const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
		const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
		const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
		pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
		pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
		pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);
		int pitch = src.GetPitch<pixel_t>(PLANAR_Y);
		int srcPitch = src.GetPitch<pixel_t>(PLANAR_U);
		int dstPitch = dst.GetPitch<pixel_t>(PLANAR_U);
		int widthUV = vi.width >> logUVx;
		int heightUV = vi.height >> logUVy;
		void(*table[2][8])(
			pixel_t* dst, const pixel_t* src,
			int width, int height, int dstPitch, int srcPitch, PNeoEnv env) =
		{
			{ // CPU
				cpu_el_from444<pixel_t, 1, 0>,
				cpu_el_from444<pixel_t, 1, 1>,
			},
			{ // CUDA
				launch_el_from444<pixel_t, 1, 0>,
				launch_el_from444<pixel_t, 1, 1>,
			}
		};
		if (IS_CUDA) {
			table[1][logUVy](dstU, srcU, widthUV, heightUV, dstPitch, srcPitch, env);
			DEBUG_SYNC;
			table[1][logUVy](dstV, srcV, widthUV, heightUV, dstPitch, srcPitch, env);
			DEBUG_SYNC;
		}
		else {
			table[0][logUVy](dstU, srcU, widthUV, heightUV, dstPitch, srcPitch, env);
			table[0][logUVy](dstV, srcV, widthUV, heightUV, dstPitch, srcPitch, env);
		}
		Copy(dstY, pitch, srcY, pitch, vi.width, vi.height, env);
		return dst.frame;
	}

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
		VideoInfo workvi = vi;
		if (uv) {
			workvi.pixel_type = Get444Type(vi);
		}

		Frame src = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(workvi);
    Frame el = (repair > 0) ? env->NewVideoFrame(workvi) : Frame();
    Frame tmp = (repair > 0) ? env->NewVideoFrame(workvi) : Frame();
		Frame srcUV = uv ? GetUVFrame<pixel_t>(src, env) : nullptr;

    const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* srcU = (!srcUV ? src : srcUV).GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* srcV = (!srcUV ? src : srcUV).GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);
    pixel_t* elY = (repair > 0) ? el.GetWritePtr<pixel_t>(PLANAR_Y) : nullptr;
    pixel_t* elU = (repair > 0) ? el.GetWritePtr<pixel_t>(PLANAR_U) : nullptr;
    pixel_t* elV = (repair > 0) ? el.GetWritePtr<pixel_t>(PLANAR_V) : nullptr;
    pixel_t* tmpY = (repair > 0) ? tmp.GetWritePtr<pixel_t>(PLANAR_Y) : nullptr;
    pixel_t* tmpU = (repair > 0) ? tmp.GetWritePtr<pixel_t>(PLANAR_U) : nullptr;
    pixel_t* tmpV = (repair > 0) ? tmp.GetWritePtr<pixel_t>(PLANAR_V) : nullptr;
    int pitchY = src.GetPitch<pixel_t>(PLANAR_Y);

    void(*table[2][8])(
			pixel_t* dstY, pixel_t* dstU, pixel_t* dstV,
			const pixel_t* srcY, const pixel_t* srcU, const pixel_t* srcV,
      int width, int height, int pitch, int maxv, int str, int strUV, int thrs, PNeoEnv env) =
    {
      { // CPU
				cpu_edgelevel<pixel_t, false, false, false>,
				cpu_edgelevel<pixel_t, false, false, true>,
				cpu_edgelevel<pixel_t, false, true, false>,
				cpu_edgelevel<pixel_t, false, true, true>,
				cpu_edgelevel<pixel_t, true, false, false>,
				cpu_edgelevel<pixel_t, true, false, true>,
				cpu_edgelevel<pixel_t, true, true, false>,
				cpu_edgelevel<pixel_t, true, true, true>,
      },
      { // CUDA
        launch_edgelevel<pixel_t, false, false, false>,
				launch_edgelevel<pixel_t, false, false, true>,
        launch_edgelevel<pixel_t, false, true, false>,
				launch_edgelevel<pixel_t, false, true, true>,
        launch_edgelevel<pixel_t, true, false, false>,
				launch_edgelevel<pixel_t, true, false, true>,
        launch_edgelevel<pixel_t, true, true, false>,
				launch_edgelevel<pixel_t, true, true, true>,
      }
    };

    int maxv = (1 << vi.BitsPerComponent()) - 1;
    int numel = max((repair + 1) / 2, 1);

    for (int i = 0; i < numel; ++i) {
      int table_idx = ((show && (i + 1 == numel)) ? 4 : 0) + (repair ? 2 : 0) + (uv ? 1 : 0);
      const pixel_t* elsrcY = srcY;
      const pixel_t* elsrcU = srcU;
      const pixel_t* elsrcV = srcV;
      if (i > 0) {
        std::swap(dst, tmp);
        std::swap(dstY, tmpY);
        std::swap(dstU, tmpU);
        std::swap(dstV, tmpV);
        elsrcY = tmpY;
        elsrcU = tmpU;
        elsrcV = tmpV;
      }
      if (IS_CUDA) {
        table[1][table_idx](dstY, dstU, dstV, elsrcY, elsrcU, elsrcV,
					vi.width, vi.height, pitchY, maxv, str, strUV, thrs, env);
        DEBUG_SYNC;
      }
      else {
        table[0][table_idx](dstY, dstU, dstV, elsrcY, elsrcU, elsrcV,
					vi.width, vi.height, pitchY, maxv, str, strUV, thrs, env);
      }
    }

    if (show) {
			ClearUV<pixel_t>(dst, env);
    }
    else {
      if (repair > 0) {
        // リペアする
        // dstをelに入れて、elをリペアで適用する
        // dstとtmpをswapしながらelを参照して適用していく
        // elは参照のみで変更しない
        std::swap(dst, el);
        std::swap(dstY, elY);
        std::swap(dstU, elU);
        std::swap(dstV, elV);
        for (int i = 0; i < repair; ++i) {
          const pixel_t* repsrcY = srcY;
          const pixel_t* repsrcU = srcU;
          const pixel_t* repsrcV = srcV;
          if (i > 0) {
            std::swap(dst, tmp);
            std::swap(dstY, tmpY);
            std::swap(dstU, tmpU);
            std::swap(dstV, tmpV);
            repsrcY = tmpY;
            repsrcU = tmpU;
            repsrcV = tmpV;
          }
          if (IS_CUDA) {
            launch_edgelevel_repair<pixel_t, 3>(dstY, elY, repsrcY, vi.width, vi.height, pitchY, env);
            DEBUG_SYNC;
            if (uv) {
              launch_edgelevel_repair<pixel_t, 3>(dstU, elU, repsrcU, vi.width, vi.height, pitchY, env);
              DEBUG_SYNC;
              launch_edgelevel_repair<pixel_t, 3>(dstV, elV, repsrcV, vi.width, vi.height, pitchY, env);
              DEBUG_SYNC;
            }
          }
          else {
            cpu_edgelevel_repair<pixel_t, 3>(dstY, elY, repsrcY, vi.width, vi.height, pitchY, env);
            if (uv) {
              cpu_edgelevel_repair<pixel_t, 3>(dstU, elU, repsrcU, vi.width, vi.height, pitchY, env);
              cpu_edgelevel_repair<pixel_t, 3>(dstV, elV, repsrcV, vi.width, vi.height, pitchY, env);
            }
          }
        }
      }

      if (uv) {
				return GetOutFrame<pixel_t>(dst, env);
      }

			CopyUV<pixel_t>(dst, src, env);
    }

		return dst.frame;
  }

public:
  KEdgeLevel(PClip clip, int str, int strUV, float thrs, int repair, bool show, PNeoEnv env)
    : KDebandBase(clip)
		, str(str)
		, strUV(strUV)
    , thrs(scaleParam(thrs, vi.BitsPerComponent()))
    , repair(repair)
    , uv(strUV > 0)
    , show(show)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KEdgeLevel] Unsupported pixel format");
      break;
    }
    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KDebandBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
		/* uvが有効なときは以下のようなスクリプトを実行
		  ポイントはBilinearResize->PointResizeが無劣化、つまり、
			途中で変更されなければ入力と同一の結果が出力されること
			これを実現するためにサンプルポイントはMPEG2のポジションではなく、少しズレている
			具体的には、MPEG2では左側の中間だが、このスクリプトでは左上をサンプルポイントとしている
			KEdgeLevelに入力されるUVは0.5ピクセルだけ上にズレることになるが、
			KEdgeLevelの処理に影響するだけで、すぐに元に戻る、かつ、KEdgeLevelでの0.5ピクセルのズレは
			さほど影響ないと思うので、大丈夫なはず
		  # -----------------------
			u = src.UToY()
			v = src.VToY()
			w = u.Width()
			h = u.Height()
			u = u.BilinearResize(w*2,h*2,0.25,0.25,w,h)
			v = v.BilinearResize(w*2,h*2,0.25,0.25,w,h)
			t = YtoUV(u, v, src).KEdgeLevel(16, 10, 2, struv=32)
			u = t.UToY().PointResize(w,h)
			v = t.VToY().PointResize(w,h)
			el1 = YtoUV(u, v, t)
		  # -----------------------
		*/

		PNeoEnv env = env_;
    return new KEdgeLevel(
			args[0].AsClip(),           // clip
			args[1].AsInt(10),          // str
			args[4].AsInt(0),           // strUV
      (float)args[2].AsFloat(25), // thrs
      args[3].AsInt(0),           // repair
      args[5].AsBool(false),      // show
      env);
  }
};


void AddFuncDebandKernel(IScriptEnvironment* env)
{
  env->AddFunction("KTemporalNR", "c[dist]i[thresh]f", KTemporalNR::Create, 0);
  env->AddFunction("KDeband", "c[range]i[thresh]f[sample]i[blur_first]b", KDeband::Create, 0);
  env->AddFunction("KEdgeLevel", "c[str]i[thrs]f[repair]i[struv]i[show]b", KEdgeLevel::Create, 0);
}

