#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "DeviceLocalData.h"

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
			CUDA_CHECK(cudaGetLastError()); \
      CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

#define IS_CUDA (env->GetProperty(AEP_DEVICE_TYPE) == DEV_TYPE_CUDA)

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
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    logUVx = vi.GetPlaneWidthSubsampling(PLANAR_U);
    logUVy = vi.GetPlaneHeightSubsampling(PLANAR_U);

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

#pragma endregion

void AddFuncKernel(IScriptEnvironment2* env)
{
	env->AddFunction("KTGMC_Bob", "c[b]f[c]f", KTGMC_Bob::Create, 0);
}
