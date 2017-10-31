
#include <stdint.h>
#include <avisynth.h>

#include <memory>
#include <algorithm>

#include "CommonFunctions.h"
#include "VectorFunctions.cuh"

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

static int clamp(int n, int min, int max) {
	return std::max(min, std::min(max, n));
}


enum {
	TEMPORAL_NR_BATCH = 4
};

template<typename vpixel_t> struct TemporalNRPtrs {
	vpixel_t* out[TEMPORAL_NR_BATCH];
	const vpixel_t* in[33];
};

static int abs_diff(int a, int b) {
	int diff = a - b;
	return (diff >= 0) ? diff : -diff;
}

void count_pixel(int4 diff, int thresh, int4& cnt)
{
	if (diff.x <= thresh) cnt.x++;
	if (diff.y <= thresh) cnt.y++;
	if (diff.z <= thresh) cnt.z++;
	if (diff.w <= thresh) cnt.w++;
}

void sum_pixel(int4 diff, int thresh, int4 ref, int4& sum)
{
	if (diff.x <= thresh) sum.x += ref.x;
	if (diff.y <= thresh) sum.y += ref.y;
	if (diff.z <= thresh) sum.z += ref.z;
	if (diff.w <= thresh) sum.w += ref.w;
}

void average_pixel(int4 sum, int4 cnt, int4& out)
{
	// CUDA”Å‚Í __fdividef ‚ðŽg‚¤
	out.x = (int)((float)sum.x / cnt.x + 0.5f);
	out.y = (int)((float)sum.y / cnt.y + 0.5f);
	out.z = (int)((float)sum.z / cnt.z + 0.5f);
	out.w = (int)((float)sum.w / cnt.w + 0.5f);
}

template<typename vpixel_t>
void cpu_temporal_nr(TemporalNRPtrs<vpixel_t>* data,
	int nframes, int mid, int width, int height, int pitch, int thresh)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			for (int b = 0; b < TEMPORAL_NR_BATCH; ++b) {
				vpixel_t center = data->in[b + mid][x + y * pitch];

				int4 pixel_count = 0;
				for (int i = 0; i < nframes; ++i) {
					vpixel_t ref = data->in[b + i][x + y * pitch];
					int4 diff = absdiff(ref, center);
					count_pixel(diff, thresh, pixel_count);
				}

				int4 sum = 0;
				for (int i = 0; i < nframes; ++i) {
					vpixel_t ref = data->in[b + i][x + y * pitch];
					int4 diff = absdiff(ref, center);
					sum_pixel(diff, thresh, to_int(ref), sum);
				}

				average_pixel(sum, pixel_count, data->out[b][x + y * pitch]);
			}
		}
	}
}

class KTemporalNR : public GenericVideoFilter
{
	typedef uint8_t pixel_t;

	enum {
		BATCH = TEMPORAL_NR_BATCH
	};

	int dist;
	int thresh;

	int logUVx;
	int logUVy;

	PVideoFrame cacheframes[BATCH];
	int cached_cycle;

	void MakeFrames(int cycle, IScriptEnvironment2* env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		int nframes = dist * 2 + 1;
		auto frames = std::unique_ptr<PVideoFrame[]>(new PVideoFrame[nframes]);
		for (int i = 0; i < nframes; ++i) {
			frames[i] = child->GetFrame(clamp(cycle * BATCH - dist + i, 0, vi.num_frames - 1), env);
		}
		
		TemporalNRPtrs<vpixel_t> ptrsY, ptrsU, ptrsV;
		for (int i = 0; i < BATCH; ++i) {
			cacheframes[i] = env->NewVideoFrame(vi);
			PVideoFrame& dst = cacheframes[i];
			ptrsY.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
			ptrsU.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
			ptrsV.out[i] = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));
		}
		for (int i = 0; i < nframes; ++i) {
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
			// TODO:
		}
		else {
			cpu_temporal_nr(&ptrsY, nframes, mid, width4, vi.height, pitchY, thresh);
			cpu_temporal_nr(&ptrsU, nframes, mid, width4UV, heightUV, pitchUV, thresh);
			cpu_temporal_nr(&ptrsV, nframes, mid, width4UV, heightUV, pitchUV, thresh);
		}
	}

public:
	KTemporalNR(PClip clip, int dist, int thresh, IScriptEnvironment* env)
		: GenericVideoFilter(clip)
		, dist(dist)
		, thresh(thresh << (vi.BitsPerComponent() - 8))
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
		, cached_cycle(-1)
	{
	}

	PVideoFrame GetFrame(int n, IScriptEnvironment* env_)
	{
		IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
		int cycle = n / BATCH;
		if (cached_cycle == cycle) {
			return cacheframes[n % BATCH];
		}
		MakeFrames(cycle, env);
		return cacheframes[n % BATCH];
	}

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_MTMODE) return MT_SERIALIZED;
		return 0;
	}

};

class KDeband : public GenericVideoFilter
{
public:
	KDeband(PClip clip, int thresh, )
};

