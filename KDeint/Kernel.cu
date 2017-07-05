#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = call; \
		if (err__ != cudaSuccess) { \
			env->ThrowError("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
				} \
		} while (0)

static int nblocks(int n, int block) {
	return (n + block - 1) / block;
}

template<typename T>
__host__ __device__ T min(T v1, T v2)
{
	return v1 < v2 ? v1 : v2;
}

template<typename T>
__host__ __device__ T max(T v1, T v2)
{
	return v1 > v2 ? v1 : v2;
}

template<typename T>
__host__ __device__ T clamp(T n, T min, T max)
{
	n = n > max ? max : n;
	return n < min ? min : n;
}

#pragma region resample

struct ResamplingProgram {
	IScriptEnvironment2 * env;
	int source_size, target_size;
	double crop_start, crop_size;
	int filter_size;

	// Array of Integer indicate starting point of sampling
	int* pixel_offset;
	int* dev_pixel_offset;

	// Array of array of coefficient for each pixel
	// {{pixel[0]_coeff}, {pixel[1]_coeff}, ...}
	float* pixel_coefficient_float;
	float* dev_pixel_coefficient_float;

	ResamplingProgram(int filter_size, int source_size, int target_size, double crop_start, double crop_size, IScriptEnvironment2* env)
		: filter_size(filter_size), source_size(source_size), target_size(target_size), crop_start(crop_start), crop_size(crop_size),
		pixel_offset(0), pixel_coefficient_float(0), env(env)
	{
		pixel_offset = (int*)env->Allocate(sizeof(int) * target_size, 64, AVS_NORMAL_ALLOC); // 64-byte alignment
		pixel_coefficient_float = (float*)env->Allocate(sizeof(float) * target_size * filter_size, 64, AVS_NORMAL_ALLOC);
		if (!pixel_offset || !pixel_coefficient_float) {
			env->Free(pixel_offset);
			env->Free(pixel_coefficient_float);
			env->ThrowError("ResamplingProgram: Could not reserve memory.");
		}
		CUDA_CHECK(cudaMalloc((void**)&dev_pixel_offset, sizeof(int) * target_size));
		CUDA_CHECK(cudaMalloc((void**)&dev_pixel_coefficient_float, sizeof(float) * target_size * filter_size));
	};

	void toCUDA() {
		CUDA_CHECK(cudaMemcpy(dev_pixel_offset, pixel_offset,
			sizeof(int) * target_size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_pixel_coefficient_float, pixel_coefficient_float,
			sizeof(float) * target_size * filter_size, cudaMemcpyHostToDevice));
	}

	~ResamplingProgram() {
		env->Free(pixel_offset);
		env->Free(pixel_coefficient_float);
		CUDA_CHECK(cudaFree(dev_pixel_offset));
		CUDA_CHECK(cudaFree(dev_pixel_coefficient_float));
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

	virtual ResamplingProgram* GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, IScriptEnvironment2* env);
};

ResamplingProgram* ResamplingFunction::GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, IScriptEnvironment2* env)
{
	double filter_scale = double(target_size) / crop_size;
	double filter_step = min(filter_scale, 1.0);
	double filter_support = support() / filter_step;
	int fir_filter_size = int(ceil(filter_support * 2));

	ResamplingProgram* program = new ResamplingProgram(fir_filter_size, source_size, target_size, crop_start, crop_size, env);

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

		program->pixel_offset[i] = start_pos;

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
			program->pixel_coefficient_float[i*fir_filter_size + k] = float(new_value - value); // no scaling for float
			value = new_value;
		}

		pos += pos_step;
	}

	program->toCUDA();

	return program;
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
			result += src[x + (begin + i) * src_pitch] * coef[x * filter_size + i];
		}
		if (!std::is_floating_point<pixel_t>::value) {  // floats are uncapped
			result = clamp<float>(result, 0, (sizeof(pixel_t) == 1) ? 255 : 65535);
		}
		dst[x + y * dst_pitch] = (pixel_t)result;
	}
}

class KDeintBob : public GenericVideoFilter {
	std::unique_ptr<ResamplingProgram> program_e_y;
	std::unique_ptr<ResamplingProgram> program_e_uv;
	std::unique_ptr<ResamplingProgram> program_o_y;
	std::unique_ptr<ResamplingProgram> program_o_uv;

	bool parity;

	// 1フレーム分キャッシュしておく
	int cacheN;
	PVideoFrame cache[2];

	void MakeFrame(bool top, PVideoFrame& src, PVideoFrame& dst,
		ResamplingProgram* program_y, ResamplingProgram* program_uv, IScriptEnvironment2* env)
	{
		for (int p = 0; p < 3; ++p) {
			const BYTE* srcptr = src->GetReadPtr(p);
			int src_pitch = src->GetPitch(p);

			// separate field
			srcptr += top ? 0 : src_pitch;
			src_pitch *= 2;

			BYTE* dstptr = dst->GetWritePtr(p);
			int dst_pitch = dst->GetPitch(p);

			ResamplingProgram* prog = (p == 0) ? program_y : program_uv;

			dim3 threads(32, 16);
			dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
			kl_resample_v<uint8_t, 4><<<blocks, threads>>>(
				(uint8_t*)srcptr, (uint8_t*)dstptr,
				src_pitch, dst_pitch, vi.width, vi.height,
				prog->dev_pixel_offset, prog->dev_pixel_coefficient_float);

			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

public:
	KDeintBob(PClip _child, IScriptEnvironment* env_)
		: GenericVideoFilter(_child)
		, parity(_child->GetParity(0))
		, cacheN(-1)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

		// フレーム数、FPSを2倍
		vi.num_frames *= 2;
		vi.fps_numerator *= 2;
		
		double shift = parity ? 0.25 : -0.25;

		double b = 0;
		double c = 0.5;

		int y_height = vi.height;
		int uv_height = vi.height / 2;

		program_e_y = std::unique_ptr<ResamplingProgram>(
			MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, shift, y_height / 2, y_height, env));
		program_e_uv = std::unique_ptr<ResamplingProgram>(
			MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, shift, uv_height / 2, uv_height, env));
		program_o_y = std::unique_ptr<ResamplingProgram>(
			MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, -shift, y_height / 2, y_height, env));
		program_o_uv = std::unique_ptr<ResamplingProgram>(
			MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, -shift, uv_height / 2, uv_height, env));
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

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

		CUDA_CHECK(cudaStreamSynchronize(NULL));

		return cache[n % 2];
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
		return new KDeintBob(args[0].AsClip(), env);
	}
};

#pragma endregion

void AddFuncKernel(IScriptEnvironment* env)
{
	env->AddFunction("KDeintBob", "c", KDeintBob::Create, 0);
}
