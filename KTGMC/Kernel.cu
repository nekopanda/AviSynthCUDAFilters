#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "DeviceLocalData.h"
#include "DebugWriter.h"
#include "CudaDebug.h"
#include "ReduceKernel.cuh"
#include "VectorFunctions.cuh"
#include "GenericImageFunctions.cuh"

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
			CUDA_CHECK(cudaGetLastError()); \
      CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

#define IS_CUDA (env->GetProperty(AEP_DEVICE_TYPE) == DEV_TYPE_CUDA)

template <typename T> struct VectorType {};

template <> struct VectorType<unsigned char> {
  typedef uchar4 type;
};

template <> struct VectorType<unsigned short> {
  typedef ushort4 type;
};

#pragma region resample

struct ResamplingProgram {
	IScriptEnvironment2 * env;
	int source_size, target_size;
	double crop_start, crop_size;
	int filter_size;

	// Array of Integer indicate starting point of sampling
	std::unique_ptr<DeviceLocalData<int>> pixel_offset;

	// Array of array of coefficient for each pixel
	// {{pixel[0]_coeff}, {pixel[1]_coeff}, ...}
  std::unique_ptr<DeviceLocalData<float>> pixel_coefficient_float;

	ResamplingProgram(int filter_size, int source_size, int target_size, double crop_start, double crop_size, 
    int* ppixel_offset, float* ppixel_coefficient_float, IScriptEnvironment2* env)
		: filter_size(filter_size), source_size(source_size), target_size(target_size), crop_start(crop_start), crop_size(crop_size),
		env(env)
	{
		pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
      new DeviceLocalData<int>(ppixel_offset, target_size, env));
    pixel_coefficient_float = std::unique_ptr<DeviceLocalData<float>>(
      new DeviceLocalData<float>(ppixel_coefficient_float, target_size * filter_size, env));
	};
};

/*********************************
*** Mitchell-Netravali filter ***
*********************************/

class ResamplingFunction
	/**
	* Pure virtual base class for resampling functions
	*/
{
public:
	virtual double f(double x) = 0;
	virtual double support() = 0;

	virtual std::unique_ptr<ResamplingProgram> GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, IScriptEnvironment2* env);
};

std::unique_ptr<ResamplingProgram> ResamplingFunction::GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, IScriptEnvironment2* env)
{
	double filter_scale = double(target_size) / crop_size;
	double filter_step = min(filter_scale, 1.0);
	double filter_support = support() / filter_step;
	int fir_filter_size = int(ceil(filter_support * 2));

  std::unique_ptr<int[]> pixel_offset = std::unique_ptr<int[]>(new int[target_size]);
  std::unique_ptr<float[]> pixel_coefficient_float = std::unique_ptr<float[]>(new float[target_size * fir_filter_size]);
	//ResamplingProgram* program = new ResamplingProgram(fir_filter_size, source_size, target_size, crop_start, crop_size, env);

	// this variable translates such that the image center remains fixed
	double pos;
	double pos_step = crop_size / target_size;

	if (source_size <= filter_support) {
		env->ThrowError("Resize: Source image too small for this resize method. Width=%d, Support=%d", source_size, int(ceil(filter_support)));
	}

	if (fir_filter_size == 1) // PointResize
		pos = crop_start;
	else
		pos = crop_start + ((crop_size - target_size) / (target_size * 2)); // TODO this look wrong, gotta check

	for (int i = 0; i < target_size; ++i) {
		// Clamp start and end position such that it does not exceed frame size
		int end_pos = int(pos + filter_support);

		if (end_pos > source_size - 1)
			end_pos = source_size - 1;

		int start_pos = end_pos - fir_filter_size + 1;

		if (start_pos < 0)
			start_pos = 0;

    pixel_offset[i] = start_pos;

		// the following code ensures that the coefficients add to exactly FPScale
		double total = 0.0;

		// Ensure that we have a valid position
		double ok_pos = clamp(pos, 0.0, (double)(source_size - 1));

		// Accumulate all coefficients for weighting
		for (int j = 0; j < fir_filter_size; ++j) {
			total += f((start_pos + j - ok_pos) * filter_step);
		}

		if (total == 0.0) {
			// Shouldn't happened for valid positions.
			total = 1.0;
		}

		double value = 0.0;

		// Now we generate real coefficient
		for (int k = 0; k < fir_filter_size; ++k) {
			double new_value = value + f((start_pos + k - ok_pos) * filter_step) / total;
      pixel_coefficient_float[i*fir_filter_size + k] = float(new_value - value); // no scaling for float
			value = new_value;
		}

		pos += pos_step;
	}

	return std::unique_ptr<ResamplingProgram>(new ResamplingProgram(
    fir_filter_size, source_size, target_size, crop_start, crop_size,
    pixel_offset.get(), pixel_coefficient_float.get(), env));
}

class MitchellNetravaliFilter : public ResamplingFunction
	/**
	* Mitchell-Netraveli filter, used in BicubicResize
	**/
{
public:
	MitchellNetravaliFilter(double b = 1. / 3., double c = 1. / 3.);
	double f(double x);
	double support() { return 2.0; }

private:
	double p0, p2, p3, q0, q1, q2, q3;
};

MitchellNetravaliFilter::MitchellNetravaliFilter(double b, double c) {
	p0 = (6. - 2.*b) / 6.;
	p2 = (-18. + 12.*b + 6.*c) / 6.;
	p3 = (12. - 9.*b - 6.*c) / 6.;
	q0 = (8.*b + 24.*c) / 6.;
	q1 = (-12.*b - 48.*c) / 6.;
	q2 = (6.*b + 30.*c) / 6.;
	q3 = (-b - 6.*c) / 6.;
}

double MitchellNetravaliFilter::f(double x) {
	x = fabs(x);
	return (x<1) ? (p0 + x*x*(p2 + x*p3)) : (x<2) ? (q0 + x*(q1 + x*(q2 + x*q3))) : 0.0;
}

template <typename pixel_t, int filter_size>
__global__ void kl_resample_v(
	const pixel_t* __restrict__ src, pixel_t* dst,
	int src_pitch, int dst_pitch,
	int width, int height,
	const int* __restrict__ offset, const float* __restrict__ coef)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int begin = offset[y];
		float result = 0;
		for (int i = 0; i < filter_size; ++i) {
			result += src[x + (begin + i) * src_pitch] * coef[y * filter_size + i];
		}
		if (!std::is_floating_point<pixel_t>::value) {  // floats are uncapped
      auto clamped = clamp<float>(result, 0, (sizeof(pixel_t) == 1) ? 255 : 65535);
			result = pixel_t(clamped + 0.5f);
		}
#if 0
    if (x == 1200 && y == 0) {
      printf("! %f\n", result);
    }
#endif
		dst[x + y * dst_pitch] = (pixel_t)result;
	}
}

template <typename pixel_t, int filter_size>
void launch_resmaple_v(
  const pixel_t* src, pixel_t* dst,
  int src_pitch, int dst_pitch,
  int width, int height,
  const int* offset, const float* coef)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_resample_v<pixel_t, filter_size> << <blocks, threads >> >(
    src, dst, src_pitch, dst_pitch, width, height, offset, coef);
}

#pragma endregion

class KTGMC_Bob : public GenericVideoFilter {
	std::unique_ptr<ResamplingProgram> program_e_y;
	std::unique_ptr<ResamplingProgram> program_e_uv;
	std::unique_ptr<ResamplingProgram> program_o_y;
	std::unique_ptr<ResamplingProgram> program_o_uv;

	bool parity;
  int logUVx;
  int logUVy;

	// 1フレーム分キャッシュしておく
	int cacheN;
	PVideoFrame cache[2];

  template <typename pixel_t>
	void MakeFrameT(bool top, PVideoFrame& src, PVideoFrame& dst,
		ResamplingProgram* program_y, ResamplingProgram* program_uv, IScriptEnvironment2* env)
	{
    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		for (int p = 0; p < 3; ++p) {
      const pixel_t* srcptr = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
			int src_pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);

			// separate field
			srcptr += top ? 0 : src_pitch;
			src_pitch *= 2;

      pixel_t* dstptr = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
			int dst_pitch = dst->GetPitch(planes[p]) / sizeof(pixel_t);

			ResamplingProgram* prog = (p == 0) ? program_y : program_uv;

      int width = vi.width;
      int height = vi.height;
      
      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      launch_resmaple_v<pixel_t, 4>(
				srcptr, dstptr, src_pitch, dst_pitch, width, height,
				prog->pixel_offset->GetData(env), prog->pixel_coefficient_float->GetData(env));

      DEBUG_SYNC;
		}
	}

  void MakeFrame(bool top, PVideoFrame& src, PVideoFrame& dst,
    ResamplingProgram* program_y, ResamplingProgram* program_uv, IScriptEnvironment2* env)
  {
    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      MakeFrameT<uint8_t>(top, src, dst, program_y, program_uv, env);
      break;
    case 2:
      MakeFrameT<uint16_t>(top, src, dst, program_y, program_uv, env);
      break;
    default:
      env->ThrowError("[KTGMC_Bob] 未対応フォーマット");
    }
  }

public:
  KTGMC_Bob(PClip _child, double b, double c, IScriptEnvironment* env_)
		: GenericVideoFilter(_child)
		, parity(_child->GetParity(0))
		, cacheN(-1)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

		// フレーム数、FPSを2倍
		vi.num_frames *= 2;
		vi.fps_numerator *= 2;
		
		double shift = parity ? 0.25 : -0.25;

		int y_height = vi.height;
		int uv_height = vi.height >> logUVy;

		program_e_y = MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, shift, y_height / 2, y_height, env);
		program_e_uv = MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, shift, uv_height / 2, uv_height, env);
		program_o_y = MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, -shift, y_height / 2, y_height, env);
		program_o_uv = MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, -shift, uv_height / 2, uv_height, env);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    if (!IS_CUDA) {
      env->ThrowError("[KTGMC_Bob] CUDAフレームを入力してください");
    }

		int srcN = n / 2;

		if (cacheN >= 0 && srcN == cacheN) {
			return cache[n % 2];
		}

		PVideoFrame src = child->GetFrame(srcN, env);

		PVideoFrame bobE = env->NewVideoFrame(vi);
		PVideoFrame bobO = env->NewVideoFrame(vi);

		MakeFrame(parity, src, bobE, program_e_y.get(), program_e_uv.get(), env);
		MakeFrame(!parity, src, bobO, program_o_y.get(), program_o_uv.get(), env);

		cacheN = n / 2;
		cache[0] = bobE;
		cache[1] = bobO;

		return cache[n % 2];
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
		return new KTGMC_Bob(
      args[0].AsClip(),
      args[1].AsFloat(0),
      args[2].AsFloat(0.5),
      env);
	}
};

enum { CALC_SAD_THREADS = 256 };

__global__ void kl_init_sad(float *sad)
{
  sad[threadIdx.x] = 0;
}

template <typename vpixel_t>
__global__ void kl_calculate_sad(const vpixel_t* pA, const vpixel_t* pB, int width4, int height, int pitch4, float* sad)
{
  int y = blockIdx.x;

  float tmpsad = 0;
  for (int x = threadIdx.x; x < width4; x += blockDim.x) {
    int4 p = absdiff(pA[x + y * pitch4], pB[x + y * pitch4]);
    tmpsad += p.x + p.y + p.z + p.w;
  }

  __shared__ float sbuf[CALC_SAD_THREADS];
  dev_reduce<float, CALC_SAD_THREADS, AddReducer<float>>(threadIdx.x, tmpsad, sbuf);

  if (threadIdx.x == 0) {
    atomicAdd(sad, tmpsad);
  }
}

template <typename vpixel_t>
__global__ void kl_binomial_temporal_soften_1(
  vpixel_t* pDst, const vpixel_t* __restrict__ pSrc,
  const vpixel_t* __restrict__ pRef0, const vpixel_t* __restrict__ pRef1,
  const float* __restrict__ sad, float scenechange, int width4, int height, int pitch4)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  __shared__ bool isSC[2];
  if (threadIdx.x < 2 && threadIdx.y == 0) {
    isSC[threadIdx.x] = (sad[threadIdx.x] >= scenechange);
  }
  __syncthreads();

  if (x < width4 && y < height) {
    int4 src = to_int(pSrc[x + y * pitch4]);
    int4 ref0 = isSC[0] ? src : to_int(pRef0[x + y * pitch4]);
    int4 ref1 = isSC[1] ? src : to_int(pRef1[x + y * pitch4]);

    int4 tmp = (ref0 + src * 2 + ref1 + 2) >> 2;
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

template <typename vpixel_t>
__global__ void kl_binomial_temporal_soften_2(
  vpixel_t* pDst, const vpixel_t* __restrict__ pSrc,
  const vpixel_t* __restrict__ pRef0, const vpixel_t* __restrict__ pRef1,
  const vpixel_t* __restrict__ pRef2, const vpixel_t* __restrict__ pRef3,
  const float* __restrict__ sad, float scenechange, int width4, int height, int pitch4)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  __shared__ bool isSC[4];
  if (threadIdx.x < 4 && threadIdx.y == 0) {
    isSC[threadIdx.x] = (sad[threadIdx.x] >= scenechange);
  }
  __syncthreads();

  if (x < width4 && y < height) {
    int4 src = to_int(pSrc[x + y * pitch4]);
    int4 ref0 = isSC[0] ? src : to_int(pRef0[x + y * pitch4]);
    int4 ref1 = isSC[1] ? src : to_int(pRef1[x + y * pitch4]);
    int4 ref2 = isSC[2] ? src : to_int(pRef2[x + y * pitch4]);
    int4 ref3 = isSC[3] ? src : to_int(pRef3[x + y * pitch4]);

    int4 tmp = (ref2 + ref0 * 4 + src * 6 + ref1 * 4 + ref3 + 4) >> 4;
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class BinomialTemporalSoften : public GenericVideoFilter {

  int radius;
  int scenechange;
  bool chroma;

  int logUVx;
  int logUVy;

  PVideoFrame GetRefFrame(int ref, IScriptEnvironment2* env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return child->GetFrame(ref, env);
  }

  template <typename pixel_t>
  PVideoFrame Proc(int n, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    PVideoFrame src = GetRefFrame(n, env);
    PVideoFrame prv1 = GetRefFrame(n - 1, env);
    PVideoFrame fwd1 = GetRefFrame(n + 1, env);
    PVideoFrame prv2, fwd2;

    if (radius >= 2) {
      prv2 = GetRefFrame(n - 2, env);
      fwd2 = GetRefFrame(n + 2, env);
    }

    PVideoFrame work;
    int work_bytes = sizeof(float) * radius * 2 * 3;
    VideoInfo workvi = VideoInfo();
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 2048;
    workvi.height = nblocks(work_bytes, vi.width * 4);
    work = env->NewVideoFrame(workvi);
    float* sad = reinterpret_cast<float*>(work->GetWritePtr());

    PVideoFrame dst = env->NewVideoFrame(vi);

    kl_init_sad << <1, radius * 2 * 3 >> > (sad);
    DEBUG_SYNC;

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    for (int p = 0; p < 3; ++p) {

      const vpixel_t* pSrc = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(planes[p]));
      vpixel_t* pDst = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (chroma == false && p > 0) {
        launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(pDst, pSrc, width4, height, pitch4);
        DEBUG_SYNC;
        continue;
      }

      const vpixel_t* pPrv1 = reinterpret_cast<const vpixel_t*>(prv1->GetReadPtr(planes[p]));
      const vpixel_t* pFwd1 = reinterpret_cast<const vpixel_t*>(fwd1->GetReadPtr(planes[p]));
      const vpixel_t* pPrv2;
      const vpixel_t* pFwd2;

      float* pSad = sad + p * radius * 2;
      kl_calculate_sad << <height, CALC_SAD_THREADS >> > (pSrc, pPrv1, width4, height, pitch4, &pSad[0]);
      DEBUG_SYNC;
      kl_calculate_sad << <height, CALC_SAD_THREADS >> > (pSrc, pFwd1, width4, height, pitch4, &pSad[1]);
      DEBUG_SYNC;

      if (radius >= 2) {
        pPrv2 = reinterpret_cast<const vpixel_t*>(prv2->GetReadPtr(planes[p]));
        pFwd2 = reinterpret_cast<const vpixel_t*>(fwd2->GetReadPtr(planes[p]));

        kl_calculate_sad << <height, CALC_SAD_THREADS >> > (pSrc, pPrv2, width4, height, pitch4, &pSad[2]);
        DEBUG_SYNC;
        kl_calculate_sad << <height, CALC_SAD_THREADS >> > (pSrc, pFwd2, width4, height, pitch4, &pSad[3]);
        DEBUG_SYNC;
      }

      //DataDebug<float> dsad(pSad, 2, env);
      //dsad.Show();


      float fsc = (float)scenechange * width * height;

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));

      switch (radius) {
      case 1:
        kl_binomial_temporal_soften_1 << <blocks, threads >> > (
          pDst, pSrc, pPrv1, pFwd1, pSad, fsc, width4, height, pitch4);
        DEBUG_SYNC;
        break;
      case 2:
        kl_binomial_temporal_soften_2 << <blocks, threads >> > (
          pDst, pSrc, pPrv1, pFwd1, pPrv2, pFwd2, pSad, fsc, width4, height, pitch4);
        DEBUG_SYNC;
        break;
      }
    }

    return dst;
  }

public:
  BinomialTemporalSoften(PClip _child, int radius, int scenechange, bool chroma, IScriptEnvironment* env_)
    : GenericVideoFilter(_child)
    , radius(radius)
    , scenechange(scenechange)
    , chroma(chroma)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    if (radius != 1 && radius != 2) {
      env->ThrowError("[BinomialTemporalSoften] radiusは1か2です");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    if (!IS_CUDA) {
      env->ThrowError("[BinomialTemporalSoften] CUDAフレームを入力してください");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[BinomialTemporalSoften] 未対応フォーマット");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new BinomialTemporalSoften(
      args[0].AsClip(),
      args[1].AsInt(),
      args[2].AsInt(0),
      args[3].AsBool(true),
      env);
  }
};

template <typename pixel_t>
__global__ void kl_copy_boarder1(
  pixel_t* pDst, const pixel_t* __restrict__ pSrc, int width, int height, int pitch
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  switch (blockIdx.y) {
  case 0: // top
    if(x < width) pDst[x] = pSrc[x];
    break;
  case 1: // left
    if (x < height) pDst[x * pitch] = pSrc[x * pitch];
    break;
  case 2: // bottom
    if (x < width) pDst[x + (height - 1) * pitch] = pSrc[x + (height - 1) * pitch];
    break;
  case 3: // right
    if (x < height) pDst[(width - 1) + x * pitch] = pSrc[(width - 1) + x * pitch];
    break;
  }
}

template <typename pixel_t, typename Horizontal, typename Vertical>
__global__ void kl_box3x3_filter(
  pixel_t* pDst, const pixel_t* __restrict__ pSrc, int width, int height, int pitch
)
{
  Horizontal horizontal;
  Vertical vertical;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < width && y < height) {
    pDst[x + y * pitch] = vertical(
      horizontal(pSrc[x - 1 + (y - 1) * pitch], pSrc[x + (y - 1) * pitch], pSrc[x + 1 + (y - 1) * pitch]),
      horizontal(pSrc[x - 1 + y * pitch], pSrc[x + y * pitch], pSrc[x + 1 + y * pitch]),
      horizontal(pSrc[x - 1 + (y + 1) * pitch], pSrc[x + (y + 1) * pitch], pSrc[x + 1 + (y + 1) * pitch]));
  }
}

struct RG11Horizontal {
  __device__ int operator()(int a, int b, int c) {
    return a + b * 2 + c;
  }
};
struct RG11Vertical {
  __device__ int operator()(int a, int b, int c) {
    return (a + b * 2 + c + 8) >> 4;
  }
};

struct RG20Horizontal {
  __device__ int operator()(int a, int b, int c) {
    return a + b + c;
  }
};
struct RG20Vertical {
  __device__ int operator()(int a, int b, int c) {
    return (a + b + c + 4) / 9;
  }
};

class KRemoveGrain : public GenericVideoFilter {
  
  int mode;
  int modeU;
  int modeV;

  int logUVx;
  int logUVy;

  template <typename pixel_t>
  PVideoFrame Proc(int n, IScriptEnvironment2* env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int modes[] = { mode, modeU, modeV };

    for (int p = 0; p < 3; ++p) {
      int mode = modes[p];
      if (mode == -1) continue;

      const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
      pixel_t* pDst = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (mode == 0) {
        launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
          (vpixel_t*)pDst, (const vpixel_t*)pSrc, width4, height, pitch4);
        DEBUG_SYNC;
        continue;
      }

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width - 2, threads.x), nblocks(height - 2, threads.y));

      switch (mode) {
      case 11:
      case 12:
        // [1 2 1] horizontal and vertical kernel blur
        kl_box3x3_filter<pixel_t, RG11Horizontal, RG11Vertical>
          << <blocks, threads >> > (pDst + pitch + 1, pSrc + pitch + 1, width - 2, height - 2, pitch);
        DEBUG_SYNC;
        break;

      case 20:
        // Averages the 9 pixels ([1 1 1] horizontal and vertical blur)
        kl_box3x3_filter<pixel_t, RG20Horizontal, RG20Vertical>
          << <blocks, threads >> > (pDst + pitch + 1, pSrc + pitch + 1, width - 2, height - 2, pitch);
        DEBUG_SYNC;
        break;

      default:
        env->ThrowError("[KRemoveGrain] Unsupported mode %d", modes[p]);
      }

      {
        dim3 threads(256);
        dim3 blocks(nblocks(max(height, width), threads.x), 4);
        kl_copy_boarder1 << <blocks, threads >> > (pDst, pSrc, width, height, pitch);
        DEBUG_SYNC;
      }

    }

    return dst;
  }

public:
  KRemoveGrain(PClip _child, int mode, int modeU, int modeV, IScriptEnvironment* env_)
    : GenericVideoFilter(_child)
    , mode(mode)
    , modeU(modeU)
    , modeV(modeV)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int modes[] = { mode, modeU, modeV };
    for (int p = 0; p < 3; ++p) {
      switch (modes[p]) {
      case -1:
      case 0:
      case 11:
      case 12:
      case 20:
        break;
      default:
        env->ThrowError("[KRemoveGrain] Unsupported mode %d", modes[p]);
      }
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    if (!IS_CUDA) {
      env->ThrowError("[KRemoveGrain] CUDAフレームを入力してください");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KRemoveGrain] 未対応フォーマット");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    int mode = args[1].AsInt(2);
    int modeU = args[2].AsInt(mode);
    int modeV = args[3].AsInt(modeU);
    return new KRemoveGrain(
      args[0].AsClip(),
      mode,
      modeU,
      modeV,
      env);
  }
};

void AddFuncKernel(IScriptEnvironment2* env)
{
  env->AddFunction("KTGMC_Bob", "c[b]f[c]f", KTGMC_Bob::Create, 0);
  env->AddFunction("BinomialTemporalSoften", "ci[scenechange]i[chroma]b", BinomialTemporalSoften::Create, 0);
  env->AddFunction("KRemoveGrain", "c[mode]i[modeU]i[modeV]i", KRemoveGrain::Create, 0);
}
