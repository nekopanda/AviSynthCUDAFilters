
#include <stdint.h>
#include <avisynth.h>

#include <memory>
#include <algorithm>

#include "DeviceLocalData.h"
#include "CommonFunctions.h"
#include "VectorFunctions.cuh"

#include "DeviceLocalData.cpp"

#ifndef NDEBUG
//#if 1
#define DEBUG_SYNC \
			CUDA_CHECK(cudaGetLastError()); \
      CUDA_CHECK(cudaDeviceSynchronize())
#else
#define DEBUG_SYNC
#endif

#define IS_CUDA (env->GetProperty(AEP_DEVICE_TYPE) == DEV_TYPE_CUDA)

int GetDeviceType(const PClip& clip);

template <typename T> struct VectorType {};

template <> struct VectorType<unsigned char> {
	typedef uchar4 type;
};

template <> struct VectorType<unsigned short> {
	typedef ushort4 type;
};

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
			return GetDeviceType(child) &
				(DEV_TYPE_CPU | DEV_TYPE_CUDA);
		}
		return 0;
	}
};

enum {
	TEMPORAL_NR_BATCH = 4,
	MAX_TEMPORAL_DIST = 16
};

template<typename vpixel_t> struct TemporalNRPtrs {
	vpixel_t* out[TEMPORAL_NR_BATCH];
	const vpixel_t* in[MAX_TEMPORAL_DIST * 2 + TEMPORAL_NR_BATCH];
};

__host__ __device__ void count_pixel(int4 diff, int thresh, int4& cnt)
{
	if (diff.x <= thresh) cnt.x++;
	if (diff.y <= thresh) cnt.y++;
	if (diff.z <= thresh) cnt.z++;
	if (diff.w <= thresh) cnt.w++;
}

template<typename vpixel_t>
__host__ __device__ void sum_pixel(int4 diff, int thresh, vpixel_t ref, int4& sum)
{
	if (diff.x <= thresh) sum.x += ref.x;
	if (diff.y <= thresh) sum.y += ref.y;
	if (diff.z <= thresh) sum.z += ref.z;
	if (diff.w <= thresh) sum.w += ref.w;
}

template<typename vpixel_t>
__host__ __device__ void average_pixel(int4 sum, int4 cnt, vpixel_t& out)
{
//#ifdef __CUDA_ARCH__
	// CUDA版は __fdividef を使う
#if 0 // あまり変わらないのでCPU版と同じにする
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
			for (int b = 0; b < TEMPORAL_NR_BATCH; ++b) {
				vpixel_t center = data->in[b + mid][x + y * pitch];

				int4 pixel_count = VHelper<int4>::make(0);
				for (int i = 0; i < nframes; ++i) {
					vpixel_t ref = data->in[b + i][x + y * pitch];
					int4 diff = absdiff(ref, center);
					count_pixel(diff, thresh, pixel_count);
				}

				int4 sum = VHelper<int4>::make(0);
				for (int i = 0; i < nframes; ++i) {
					vpixel_t ref = data->in[b + i][x + y * pitch];
					int4 diff = absdiff(ref, center);
					sum_pixel(diff, thresh, ref, sum);
				}

				average_pixel(sum, pixel_count, data->out[b][x + y * pitch]);
			}
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
	int b = threadIdx.y;
	int y = blockIdx.y;

	extern __shared__ void* s__[];
	vpixel_t(*sbuf)[32] = (vpixel_t(*)[32])s__;

	// pixel_cacheにデータを入れる
	if (x < width) {
		for (int i = b; i < nframes + TEMPORAL_NR_BATCH - 1; i += blockDim.y) {
			sbuf[i][tx] = data->in[i][x + pitch * y];
		}
	}

	__syncthreads();

	if (x < width) {
		vpixel_t center = sbuf[b + mid][tx];

		int4 pixel_count = VHelper<int4>::make(0);
		for (int i = 0; i < nframes; ++i) {
			vpixel_t ref = sbuf[b + i][tx];
			int4 diff = absdiff(ref, center);
			count_pixel(diff, thresh, pixel_count);
		}

		int4 sum = VHelper<int4>::make(0);
		for (int i = 0; i < nframes; ++i) {
			vpixel_t ref = sbuf[b + i][tx];
			int4 diff = absdiff(ref, center);
			sum_pixel(diff, thresh, ref, sum);
		}

		average_pixel(sum, pixel_count, data->out[b][x + y * pitch]);
	}
}

class KTemporalNR : public KDebandBase
{
	enum {
		BATCH = TEMPORAL_NR_BATCH
	};

	int dist;
	int thresh;

	PVideoFrame cacheframes[BATCH];
	int cached_cycle;

	template <typename pixel_t>
	void MakeFrames(int cycle, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		int nframes = dist * 2 + 1;
		int batchframes = nframes + BATCH - 1;
		auto frames = std::unique_ptr<PVideoFrame[]>(new PVideoFrame[batchframes]);
		for (int i = 0; i < batchframes; ++i) {
			frames[i] = child->GetFrame(clamp(cycle * BATCH - dist + i, 0, vi.num_frames - 1), env);
		}

		TemporalNRPtrs<vpixel_t> *ptrs = new TemporalNRPtrs<vpixel_t>[3];
		TemporalNRPtrs<vpixel_t>& ptrsY = ptrs[0];
		TemporalNRPtrs<vpixel_t>& ptrsU = ptrs[1];
		TemporalNRPtrs<vpixel_t>& ptrsV = ptrs[2];

		for (int i = 0; i < BATCH; ++i) {
			cacheframes[i] = env->NewVideoFrame(vi);
			PVideoFrame& dst = cacheframes[i];
			ptrsY.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
			ptrsU.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
			ptrsV.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));
		}
		for (int i = 0; i < batchframes; ++i) {
			ptrsY.in[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_Y));
			ptrsU.in[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_U));
			ptrsV.in[i] = reinterpret_cast<const vpixel_t*>(frames[i]->GetReadPtr(PLANAR_V));
		}

		int pitchY = cacheframes[0]->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = cacheframes[0]->GetPitch(PLANAR_U) / sizeof(vpixel_t);
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
			PVideoFrame work = env->NewVideoFrame(workvi);

			TemporalNRPtrs<vpixel_t>* dptrs = 
				reinterpret_cast<TemporalNRPtrs<vpixel_t>*>(work->GetWritePtr());
			CUDA_CHECK(cudaMemcpyAsync(dptrs, ptrs, work_bytes, cudaMemcpyHostToDevice));

			dim3 threads(32, TEMPORAL_NR_BATCH);
			dim3 blocks(nblocks(width4, threads.x), vi.height);
			dim3 blocksUV(nblocks(width4UV, threads.x), heightUV);
			int sbufsize = batchframes * sizeof(vpixel_t) * 32;
			kl_temporal_nr << <blocks, threads, sbufsize >> >(
				&dptrs[0], nframes, mid, width4, vi.height, pitchY, thresh);
			DEBUG_SYNC;
			kl_temporal_nr << <blocksUV, threads, sbufsize >> >(
				&dptrs[1], nframes, mid, width4UV, heightUV, pitchUV, thresh);
			DEBUG_SYNC;
			kl_temporal_nr << <blocksUV, threads, sbufsize >> >(
				&dptrs[2], nframes, mid, width4UV, heightUV, pitchUV, thresh);
			DEBUG_SYNC;

			// 終わったら解放するコールバックを追加
			static_cast<IScriptEnvironment2*>(env)->DeviceAddCallback([](void* arg) {
				delete[]((TemporalNRPtrs<vpixel_t>*)arg);
			}, ptrs);
		}
		else {
			cpu_temporal_nr(&ptrsY, nframes, mid, width4, vi.height, pitchY, thresh);
			cpu_temporal_nr(&ptrsU, nframes, mid, width4UV, heightUV, pitchUV, thresh);
			cpu_temporal_nr(&ptrsV, nframes, mid, width4UV, heightUV, pitchUV, thresh);
			delete [] ptrs;
		}
	}

public:
	KTemporalNR(PClip clip, int dist, float thresh, IScriptEnvironment* env)
		: KDebandBase(clip)
		, dist(dist)
		, thresh(scaleParam(thresh, vi.BitsPerComponent()))
		, cached_cycle(-1)
	{
		if (dist > MAX_TEMPORAL_DIST) {
			env->ThrowError("[KTemporalNR] maximum dist is 16");
		}
	}

	PVideoFrame GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
		int cycle = n / BATCH;
		if (cached_cycle == cycle) {
			return cacheframes[n % BATCH];
		}
		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			MakeFrames<uint8_t>(cycle, env);
			break;
		case 2:
			MakeFrames<uint16_t>(cycle, env);
			break;
		default:
			env->ThrowError("[KTemporalNR] Unsupported pixel format");
			break;
		}
		cached_cycle = cycle;
		return cacheframes[n % BATCH];
	}

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_MTMODE) return MT_SERIALIZED;
		return KDebandBase::SetCacheHints(cachehints, frame_range);
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
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

static __device__ __host__ int absdiff(int a, int b) {
	int r = a - b;
	return (r >= 0) ? r : -r;
}

template <typename pixel_t, int sample_mode, bool blur_first>
void cpu_reduce_banding(
	pixel_t* dst, const pixel_t* src, const uint8_t* rand,
	int width, int height, int pitch, int range, int thresh)
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
	int width, int height, int pitch, int range, int thresh)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
	kl_reduce_banding<pixel_t, sample_mode, blur_first> << <blocks, threads >> > (
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

	DeviceLocalData<uint8_t>* CreateDebandRandom(int width, int height, int seed, IScriptEnvironment2* env)
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
	PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
	{
		PVideoFrame src = child->GetFrame(n, env);
		PVideoFrame dst = env->NewVideoFrame(vi);

		const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
		const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
		const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
		int widthUV = vi.width >> logUVx;
		int heightUV = vi.height >> logUVy;

		const uint8_t* prand = rand->GetData(env);

		void (*table[2][6])(
			pixel_t* dst, const pixel_t* src, const uint8_t* rand,
			int width, int height, int pitch, int range, int thresh) =
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
			table[1][table_idx](dstY, srcY, prand, vi.width, vi.height, pitchY, range, thresh);
			DEBUG_SYNC;
			table[1][table_idx](dstU, srcU, prand, widthUV, heightUV, pitchUV, range, thresh);
			DEBUG_SYNC;
			table[1][table_idx](dstV, srcV, prand, widthUV, heightUV, pitchUV, range, thresh);
			DEBUG_SYNC;
		}
		else {
			table[0][table_idx](dstY, srcY, prand, vi.width, vi.height, pitchY, range, thresh);
			table[0][table_idx](dstU, srcU, prand, widthUV, heightUV, pitchUV, range, thresh);
			table[0][table_idx](dstV, srcV, prand, widthUV, heightUV, pitchUV, range, thresh);
		}

		return dst;
	}

public:
	KDeband(PClip clip, int range, float thresh, int sample_mode, bool blur_first, IScriptEnvironment2* env)
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

	PVideoFrame GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
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
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
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
void cpu_fill (pixel_t* dst, pixel_t v, int width, int height, int pitch)
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

template <typename pixel_t, bool check, bool selective>
void cpu_edgelevel(
	pixel_t* dsttop, const pixel_t* srctop,
	int width, int height, int pitch, int maxv, int str, int thrs)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const pixel_t* src = srctop + y * pitch + x;
			pixel_t* dst = dsttop + y * pitch + x;

			if (y <= 1 || y >= height - 2 || x <= 1 || x >= width - 2) {
				if (x < width && y < height) {
					*dst = check ? SCALE(EDGE_CHECK_NONE) : *src;
				}
			}
			else {
				int srcv = *src;
				int dstv;

				int hmax, hmin, vmax, vmin, avg;
				int hdiffmax = 0, vdiffmax = 0;
				int hprev = hmax = hmin = src[-2];
				int vprev = vmax = vmin = src[-2 * pitch];

				for (int i = -1; i < 3; ++i) {
					int hcur = src[i];
					int vcur = src[i*pitch];

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
					hdiffmax = vdiffmax;
				}

				float factor = selective ? clamp((0.5f - hdiffmax / (float)(hmax - hmin)) * 10.0f, 0.0f, 1.0f) : 1.0f;

				if (check) {
					if (hmax - hmin > thrs && factor > 0.0f) {
						avg = (hmax + hmin) >> 1;
						if (srcv > avg) {
							dstv = (factor == 1.0f) ? SCALE(EDGE_CHECK_WHITE) : SCALE(EDGE_CHECK_BRIGHT);
						}
						else {
							dstv = (factor == 1.0f) ? SCALE(EDGE_CHECK_BLACK) : SCALE(EDGE_CHECK_DARK);
						}
					}
					else {
						dstv = SCALE(EDGE_CHECK_NONE);
					}
				}
				else {
					if (hmax - hmin > thrs) {
						avg = (hmin + hmax) >> 1;

						dstv = min(max(srcv + (int)((srcv - avg) * (str * factor) * 0.0625f), hmin), hmax);
						dstv = clamp(dstv, 0, maxv);
					}
					else {
						dstv = srcv;
					}
				}

				*dst = dstv;
			}
		}
	}
}

template <typename pixel_t, bool check, bool selective>
__global__ void kl_edgelevel(
	pixel_t* __restrict__ dst, const pixel_t* __restrict__ src,
	int width, int height, int pitch, int maxv, int str, int thrs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	src += y * pitch + x;
	dst += y * pitch + x;

	if (y <= 1 || y >= height - 2 || x <= 1 || x >= width - 2) {
		if (x < width && y < height) {
			*dst = check ? SCALE(EDGE_CHECK_NONE) : *src;
		}
	}
	else {
		int srcv = *src;
		int dstv;

		int hmax, hmin, vmax, vmin, avg;
		int hdiffmax = 0, vdiffmax = 0;
		int hprev = hmax = hmin = src[-2];
		int vprev = vmax = vmin = src[-2 * pitch];

		for (int i = -1; i < 3; ++i) {
			int hcur = src[i];
			int vcur = src[i*pitch];

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
			hdiffmax = vdiffmax;
		}

		float factor = selective ? clamp((0.5f - hdiffmax / (float)(hmax - hmin)) * 10.0f, 0.0f, 1.0f) : 1.0f;

		if (check) {
			if (hmax - hmin > thrs && factor > 0.0f) {
				avg = (hmax + hmin) >> 1;
				if (srcv > avg) {
					dstv = (factor == 1.0f) ? SCALE(EDGE_CHECK_WHITE) : SCALE(EDGE_CHECK_BRIGHT);
				}
				else {
					dstv = (factor == 1.0f) ? SCALE(EDGE_CHECK_BLACK) : SCALE(EDGE_CHECK_DARK);
				}
			}
			else {
				dstv = SCALE(EDGE_CHECK_NONE);
			}
		}
		else {
			if (hmax - hmin > thrs) {
				avg = (hmin + hmax) >> 1;

				dstv = min(max(srcv + (int)((srcv - avg) * (str * factor) * 0.0625f), hmin), hmax);
				dstv = clamp(dstv, 0, maxv);
			}
			else {
				dstv = srcv;
			}
		}

		*dst = dstv;
	}
}

template <typename pixel_t, bool check, bool selective>
void launch_edgelevel(
	pixel_t* dsttop, const pixel_t* srctop,
	int width, int height, int pitch, int maxv, int str, int thrs)
{
	dim3 threads(32, 16);
	dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
	kl_edgelevel<pixel_t, check, selective> << <blocks, threads >> > (
		dsttop, srctop, width, height, pitch, maxv, str, thrs);
}

class KEdgeLevel : public KDebandBase
{
	int str;
	int thrs;
	bool selective;
	bool show;

	template <typename pixel_t>
	void CopyUV(PVideoFrame& dst, PVideoFrame& src, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
		const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_copy << <blocksUV, threads >> >(dstU, srcU, width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
			kl_copy << <blocksUV, threads >> >(dstV, srcV, width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
		}
		else {
			cpu_copy<vpixel_t>(dstU, srcU, width4UV, heightUV, pitchUV);
			cpu_copy<vpixel_t>(dstV, srcV, width4UV, heightUV, pitchUV);
		}
	}

	template <typename pixel_t>
	void ClearUV(PVideoFrame& dst, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchUV = dst->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;

		vpixel_t zerov = VHelper<vpixel_t>::make(1 << (vi.BitsPerComponent() - 1));

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_fill << <blocksUV, threads >> >(dstU, zerov, width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
			kl_fill << <blocksUV, threads >> >(dstV, zerov, width4UV, heightUV, pitchUV);
			DEBUG_SYNC;
		}
		else {
			cpu_fill<vpixel_t>(dstU, zerov, width4UV, heightUV, pitchUV);
			cpu_fill<vpixel_t>(dstV, zerov, width4UV, heightUV, pitchUV);
		}
	}

	template <typename pixel_t>
	PVideoFrame GetFrameT(int n, IScriptEnvironment2* env)
	{
		PVideoFrame src = child->GetFrame(n, env);
		PVideoFrame dst = env->NewVideoFrame(vi);

		const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);

		void(*table[2][4])(
			pixel_t* dsttop, const pixel_t* srctop,
			int width, int height, int pitch, int maxv, int str, int thrs) =
		{
			{
				cpu_edgelevel<pixel_t, false, false>,
				cpu_edgelevel<pixel_t, false, true>,
				cpu_edgelevel<pixel_t, true, false>,
				cpu_edgelevel<pixel_t, true, true>,
			},
			{
				launch_edgelevel<pixel_t, false, false>,
				launch_edgelevel<pixel_t, false, true>,
				launch_edgelevel<pixel_t, true, false>,
				launch_edgelevel<pixel_t, true, true>,
			}
		};

		int table_idx = (show ? 2 : 0) + (selective ? 1 : 0);
		int maxv = (1 << vi.BitsPerComponent()) - 1;

		if (IS_CUDA) {
			table[1][table_idx](dstY, srcY, vi.width, vi.height, pitchY, maxv, str, thrs);
			DEBUG_SYNC;
		}
		else {
			table[0][table_idx](dstY, srcY, vi.width, vi.height, pitchY, maxv, str, thrs);
		}

		if (show) {
			ClearUV<pixel_t>(dst, env);
		}
		else {
			CopyUV<pixel_t>(dst, src, env);
		}

		return dst;
	}

public:
	KEdgeLevel(PClip clip, int str, float thrs, bool selective, bool show, IScriptEnvironment2* env)
		: KDebandBase(clip)
		, str(str)
		, thrs(scaleParam(thrs, vi.BitsPerComponent()))
		, selective(selective)
		, show(show)
	{
	}

	PVideoFrame GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
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

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
		return new KEdgeLevel(
			args[0].AsClip(),           // clip
			args[1].AsInt(10),          // str
			(float)args[2].AsFloat(25), // thrs
			args[3].AsBool(false),      // selective
			args[4].AsBool(false),      // show
			env);
	}
};


void AddFuncDebandKernel(IScriptEnvironment* env)
{
	env->AddFunction("KTemporalNR", "c[dist]i[thresh]f", KTemporalNR::Create, 0);
	env->AddFunction("KDeband", "c[range]i[thresh]f[sample]i[blur_first]b", KDeband::Create, 0);
	env->AddFunction("KEdgeLevel", "c[str]i[thrs]f[selective]b[show]b", KEdgeLevel::Create, 0);
}

