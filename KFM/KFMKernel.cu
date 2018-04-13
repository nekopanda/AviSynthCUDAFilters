
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"

bool cpu_contains_durty_block(const uint8_t* flagp, int fpitch, int nBlkX, int nBlkY, int* work, int thresh)
{
	for (int by = 0; by < nBlkY; ++by) {
		for (int bx = 0; bx < nBlkX; ++bx) {
			if (flagp[bx + by * fpitch] >= thresh) return true;
		}
	}
	return false;
}

__global__ void kl_init_contains_durty_block(int* work)
{
	*work = 0;
}

__global__ void kl_contains_durty_block(const uint8_t* flagp, int fpitch, int nBlkX, int nBlkY, int* work, int thresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		if (flagp[x + y * fpitch] >= thresh) {
			*work = 1;
		}
	}
}

void cpu_binary_flag(
	uint8_t* dst, int dpitch, const uint8_t* src, int spitch, 
	int nBlkX, int nBlkY, int thresh)
{
	for (int y = 0; y < nBlkY; ++y) {
		for (int x = 0; x < nBlkX; ++x) {
			dst[x + y * dpitch] = ((src[x + y * spitch] >= thresh) ? 128 : 0);
		}
	}
}

__global__ void kl_binary_flag(
	uint8_t* dst, int dpitch, const uint8_t* src, int spitch, 
	int nBlkX, int nBlkY, int thresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nBlkX && y < nBlkY) {
		dst[x + y * dpitch] = ((src[x + y * spitch] >= thresh) ? 128 : 0);
	}
}

void cpu_bilinear_x8_v(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int y0 = ((y - 4) >> 3);
			int c0 = ((y0 + 1) << 3) - (y - 4);
			int c1 = 8 - c0;
			auto s0 = src[x + (y0 + 0) * spitch];
			auto s1 = src[x + (y0 + 1) * spitch];
			dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
		}
	}
}

__global__ void kl_bilinear_x8_v(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int y0 = ((y - 4) >> 3);
		int c0 = ((y0 + 1) << 3) - (y - 4);
		int c1 = 8 - c0;
		auto s0 = src[x + (y0 + 0) * spitch];
		auto s1 = src[x + (y0 + 1) * spitch];
		dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
	}
}

void cpu_bilinear_x8_h(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int x0 = ((x - 4) >> 3);
			int c0 = ((x0 + 1) << 3) - (x - 4);
			int c1 = 8 - c0;
			auto s0 = src[(x0 + 0) + y * spitch];
			auto s1 = src[(x0 + 1) + y * spitch];
			dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
		}
	}
}

__global__ void kl_bilinear_x8_h(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int x0 = ((x - 4) >> 3);
		int c0 = ((x0 + 1) << 3) - (x - 4);
		int c1 = 8 - c0;
		auto s0 = src[(x0 + 0) + y * spitch];
		auto s1 = src[(x0 + 1) + y * spitch];
		dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + 4) >> 3;
	}
}

template <typename vpixel_t, typename fpixel_t>
void cpu_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, 
	int width, int height, int pitch, 
	const fpixel_t* flagp, int fpitch,
	int logx, int logy, int nBlkX, int nBlkY)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int4 combe = to_int(flagp[(x << logx) + (y << logy) * fpitch]);
			int4 invcombe = VHelper<int4>::make(128) - combe;
			int4 tmp = (combe * to_int(src60[x + y * pitch]) + invcombe * to_int(src24[x + y * pitch]) + 64) >> 7;
			dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
		}
	}
}

template <typename vpixel_t, typename fpixel_t>
__global__ void kl_merge(vpixel_t* dst,
	const vpixel_t* src24, const vpixel_t* src60, 
	int width, int height, int pitch,
	const fpixel_t* flagp, int fpitch,
	int logx, int logy, int nBlkX, int nBlkY)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		int4 combe = to_int(flagp[(x << logx) + (y << logy) * fpitch]);
		int4 invcombe = VHelper<int4>::make(128) - combe;
		int4 tmp = (combe * to_int(src60[x + y * pitch]) + invcombe * to_int(src24[x + y * pitch]) + 64) >> 7;
		dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
	}
}

enum KFMSWTICH_FLAG {
	FRAME_60 = 1,
	FRAME_24,
};

class KFMSwitch : public KFMFilterBase
{
	typedef uint8_t pixel_t;

	PClip clip24;
	PClip fmclip;
	PClip combeclip;
	float thswitch;
	float thpatch;
	bool show;
	bool showflag;

	int logUVx;
	int logUVy;
	int nBlkX, nBlkY;

	VideoInfo workvi;

	PulldownPatterns patterns;

	bool ContainsDurtyBlock(PVideoFrame& flag, PVideoFrame& work, int thpatch, PNeoEnv env)
	{
		const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());
		int* pwork = reinterpret_cast<int*>(work->GetWritePtr());
		int pitch = flag->GetPitch();

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
			kl_init_contains_durty_block << <1, 1 >> > (pwork);
			kl_contains_durty_block << <blocks, threads >> > (flagp, pitch, nBlkX, nBlkY, pwork, thpatch);
			int result;
			CUDA_CHECK(cudaMemcpy(&result, pwork, sizeof(int), cudaMemcpyDeviceToHost));
			return result != 0;
		}
		else {
			return cpu_contains_durty_block(flagp, pitch, nBlkX, nBlkY, pwork, thpatch);
		}
	}

	void MakeMergeFlag(PVideoFrame& dst, PVideoFrame& src, PVideoFrame& dsttmp, PVideoFrame& srctmp, int thpatch, PNeoEnv env)
	{
		const uint8_t* srcp = reinterpret_cast<const uint8_t*>(src->GetReadPtr());
		uint8_t* dstp = reinterpret_cast<uint8_t*>(dst->GetWritePtr());
		uint8_t* dsttmpp = reinterpret_cast<uint8_t*>(dsttmp->GetWritePtr()) + dsttmp->GetPitch();
		uint8_t* srctmpp = reinterpret_cast<uint8_t*>(srctmp->GetWritePtr());

		// 0と128の2値にした後、線形補間で画像サイズまで拡大 //

		if (IS_CUDA) {
			dim3 threads(32, 8);
			dim3 binary_blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
			kl_binary_flag << <binary_blocks, threads >> >(
				srctmpp, srctmp->GetPitch(), srcp, src->GetPitch(), nBlkX, nBlkY, thpatch);
			DEBUG_SYNC;
			{
				dim3 threads(32, 1);
				dim3 blocks(nblocks(nBlkX, threads.x));
				kl_padv << <blocks, threads >> > (srctmpp, nBlkX, nBlkY, srctmp->GetPitch(), 1);
				DEBUG_SYNC;
			}
			{
				dim3 threads(1, 32);
				dim3 blocks(1, nblocks(nBlkY, threads.y));
				kl_padh << <blocks, threads >> > (srctmpp, nBlkX, nBlkY + 1 * 2, srctmp->GetPitch(), 1);
				DEBUG_SYNC;
			}
			dim3 h_blocks(nblocks(vi.width, threads.x), nblocks(nBlkY, threads.y));
			kl_bilinear_x8_h << <h_blocks, threads >> >(
				dsttmpp, vi.width, nBlkY + 2, dsttmp->GetPitch(), srctmpp - srctmp->GetPitch(), srctmp->GetPitch());
			DEBUG_SYNC;
			dim3 v_blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
			kl_bilinear_x8_v << <v_blocks, threads >> >(
				dstp, vi.width, vi.height, dst->GetPitch(), dsttmpp + dsttmp->GetPitch(), dsttmp->GetPitch());
			DEBUG_SYNC;
		}
		else {
			cpu_binary_flag(srctmpp, srctmp->GetPitch(), srcp, src->GetPitch(), nBlkX, nBlkY, thpatch);
			cpu_padv(srctmpp, nBlkX, nBlkY, srctmp->GetPitch(), 1);
			cpu_padh(srctmpp, nBlkX, nBlkY + 1 * 2, srctmp->GetPitch(), 1);
			// 上下パディング1行分も含めて処理
			cpu_bilinear_x8_h(dsttmpp, vi.width, nBlkY + 2, dsttmp->GetPitch(), srctmpp - srctmp->GetPitch(), srctmp->GetPitch());
			// ソースはパディング1行分をスキップして渡す
			cpu_bilinear_x8_v(dstp, vi.width, vi.height, dst->GetPitch(), dsttmpp + dsttmp->GetPitch(), dsttmp->GetPitch());
		}
	}

	template <typename pixel_t>
	void MergeBlock(PVideoFrame& src24, PVideoFrame& src60, PVideoFrame& flag, PVideoFrame& dst, PNeoEnv env)
	{
		typedef typename VectorType<pixel_t>::type vpixel_t;
		const vpixel_t* src24Y = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_Y));
		const vpixel_t* src24U = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_U));
		const vpixel_t* src24V = reinterpret_cast<const vpixel_t*>(src24->GetReadPtr(PLANAR_V));
		const vpixel_t* src60Y = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_Y));
		const vpixel_t* src60U = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_U));
		const vpixel_t* src60V = reinterpret_cast<const vpixel_t*>(src60->GetReadPtr(PLANAR_V));
		vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
		vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
		vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));
		const uchar4* flagp = reinterpret_cast<const uchar4*>(flag->GetReadPtr());

		int pitchY = src24->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
		int pitchUV = src24->GetPitch(PLANAR_U) / sizeof(vpixel_t);
		int width4 = vi.width >> 2;
		int width4UV = width4 >> logUVx;
		int heightUV = vi.height >> logUVy;
		int fpitch4 = flag->GetPitch() / sizeof(uchar4);

		if (IS_CUDA) {
			dim3 threads(32, 16);
			dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
			dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
			kl_merge << <blocks, threads >> >(
				dstY, src24Y, src60Y, width4, vi.height, pitchY, flagp, fpitch4, 0, 0, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
			kl_merge << <blocksUV, threads >> >(
				dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			DEBUG_SYNC;
		}
		else {
			cpu_merge(dstY, src24Y, src60Y, width4, vi.height, pitchY, flagp, fpitch4, 0, 0, nBlkX, nBlkY);
			cpu_merge(dstU, src24U, src60U, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
			cpu_merge(dstV, src24V, src60V, width4UV, heightUV, pitchUV, flagp, fpitch4, logUVx, logUVy, nBlkX, nBlkY);
		}
	}

	template <typename pixel_t>
	void VisualizeFlag(PVideoFrame& dst, PVideoFrame& mf, PNeoEnv env)
	{
		// 判定結果を表示
		int blue[] = { 73, 230, 111 };

		const uint8_t* mfp = reinterpret_cast<const uint8_t*>(mf->GetReadPtr());
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int mfpitch = mf->GetPitch(PLANAR_Y) / sizeof(uint8_t);
		int dstPitchY = dst->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int dstPitchUV = dst->GetPitch(PLANAR_U) / sizeof(pixel_t);

		// 色を付ける
		for (int y = 0; y < vi.height; ++y) {
			for (int x = 0; x < vi.width; ++x) {
				int score = mfp[x + y * mfpitch];
				int offY = x + y * dstPitchY;
				int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
				dstY[offY] = (blue[0] * score + dstY[offY] * (128 - score)) >> 7;
				dstU[offUV] = (blue[1] * score + dstU[offUV] * (128 - score)) >> 7;
				dstV[offUV] = (blue[2] * score + dstV[offUV] * (128 - score)) >> 7;
			}
		}
	}

	template <typename pixel_t>
	PVideoFrame InternalGetFrame(int n60, PVideoFrame& fmframe, int& type, PNeoEnv env)
	{
		int cycleIndex = n60 / 10;
		int kfmPattern = (int)fmframe->GetProperty("KFM_Pattern")->GetInt();
		float kfmCost = (float)fmframe->GetProperty("KFM_Cost")->GetFloat();

		if (kfmCost > thswitch || PulldownPatterns::Is30p(kfmPattern)) {
			// コストが高いので60pと判断 or 30pの場合
			PVideoFrame frame60 = child->GetFrame(n60, env);
			type = FRAME_60;
			return frame60;
		}

		type = FRAME_24;

		// 24pフレーム番号を取得
		Frame24Info frameInfo = patterns.GetFrame60(kfmPattern, n60);
		int n24 = frameInfo.cycleIndex * 4 + frameInfo.frameIndex;

		if (frameInfo.frameIndex < 0) {
			// 前に空きがあるので前のサイクル
			n24 = frameInfo.cycleIndex * 4 - 1;
		}
		else if (frameInfo.frameIndex >= 4) {
			// 後ろのサイクルのパターンを取得
			PVideoFrame nextfmframe = fmclip->GetFrame(cycleIndex + 1, env);
			int nextPattern = (int)nextfmframe->GetProperty("KFM_Pattern")->GetInt();
			int fstart = patterns.GetFrame24(nextPattern, 0).fieldStartIndex;
			if (fstart > 0) {
				// 前に空きがあるので前のサイクル
				n24 = frameInfo.cycleIndex * 4 + 3;
			}
			else {
				// 前に空きがないので後ろのサイクル
				n24 = frameInfo.cycleIndex * 4 + 4;
			}
		}

		PVideoFrame frame24 = clip24->GetFrame(n24, env);
		PVideoFrame flag = combeclip->GetFrame(n24, env)->GetProperty(COMBE_FLAG_STR)->GetFrame();

		{
			PVideoFrame work = env->NewVideoFrame(workvi);
			if (ContainsDurtyBlock(flag, work, (int)thpatch, env) == false) {
				// ダメなブロックはないのでそのまま返す
				return frame24;
			}
		}

		PVideoFrame frame60 = child->GetFrame(n60, env);

		VideoInfo mfvi = vi;
		mfvi.pixel_type = VideoInfo::CS_Y8;
		PVideoFrame mflag = env->NewVideoFrame(mfvi);

		{
			// マージ用フラグ作成
			PVideoFrame mflagtmp = env->NewVideoFrame(mfvi);
			PVideoFrame flagtmp = NewSwitchFlagFrame(vi, env->GetProperty(AEP_FRAME_ALIGN), 2, env);
			MakeMergeFlag(mflag, flag, mflagtmp, flagtmp, (int)thpatch, env);
		}

		if (!IS_CUDA && vi.ComponentSize() == 1 && showflag) {
			env->MakeWritable(&frame24);
			VisualizeFlag<pixel_t>(frame24, mflag, env);
			return frame24;
		}

		// ダメなブロックは60pフレームからコピー
		PVideoFrame dst = env->NewVideoFrame(vi);
		MergeBlock<pixel_t>(frame24, frame60, mflag, dst, env);

		return dst;
	}

	void DrawInfo(PVideoFrame& dst, const char* fps, int pattern, float score, IScriptEnvironment* env) {
		env->MakeWritable(&dst);

		char buf[100]; sprintf(buf, "KFMSwitch: %s pattern:%2d cost:%.1f", fps, pattern, score);
		DrawText(dst, true, 0, 0, buf);
	}

public:
	KFMSwitch(PClip clip60, PClip clip24, PClip fmclip, PClip combeclip,
		float thswitch, float thpatch, bool show, bool showflag, IScriptEnvironment* env)
		: KFMFilterBase(clip60)
		, clip24(clip24)
		, fmclip(fmclip)
		, combeclip(combeclip)
		, thswitch(thswitch)
		, thpatch(thpatch)
		, show(show)
		, showflag(showflag)
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{
		if (vi.width & 7) env->ThrowError("[KFMSwitch]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KFMSwitch]: height must be multiple of 8");

		nBlkX = nblocks(vi.width, OVERLAP);
		nBlkY = nblocks(vi.height, OVERLAP);

		int work_bytes = sizeof(int);
		workvi.pixel_type = VideoInfo::CS_BGR32;
		workvi.width = 4;
		workvi.height = nblocks(work_bytes, workvi.width * 4);
	}

	PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
	{
		PNeoEnv env = env_;

		int cycleIndex = n60 / 10;
		PVideoFrame fmframe = fmclip->GetFrame(cycleIndex, env);
		int frameType;

		PVideoFrame dst;
		int pixelSize = vi.ComponentSize();
		switch (pixelSize) {
		case 1:
			dst = InternalGetFrame<uint8_t>(n60, fmframe, frameType, env);
			break;
		case 2:
			dst = InternalGetFrame<uint16_t>(n60, fmframe, frameType, env);
			break;
		default:
			env->ThrowError("[KFMSwitch] Unsupported pixel format");
			break;
		}

		if (!IS_CUDA && pixelSize == 1 && show) {
			const std::pair<int, float>* pfm = (std::pair<int, float>*)fmframe->GetReadPtr();
			const char* fps = (frameType == FRAME_60) ? "60p" : "24p";
			DrawInfo(dst, fps, pfm->first, pfm->second, env);
		}

		return dst;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KFMSwitch(
			args[0].AsClip(),           // clip60
			args[1].AsClip(),           // clip24

			args[2].AsClip(),           // fmclip
			args[3].AsClip(),           // combeclip
			(float)args[4].AsFloat(0.8f),// thswitch
			(float)args[5].AsFloat(40.0f),// thpatch
			args[6].AsBool(false),      // show
			args[7].AsBool(false),      // showflag
			env
			);
	}
};

class AssertOnCUDA : public GenericVideoFilter
{
public:
	AssertOnCUDA(PClip clip) : GenericVideoFilter(clip) { }

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_DEV_TYPE) {
			return DEV_TYPE_CUDA;
		}
		return 0;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new AssertOnCUDA(args[0].AsClip());
	}
};

void AddFuncFMKernel(IScriptEnvironment* env)
{
	env->AddFunction("KFMSwitch", "cccc[thswitch]f[thpatch]f[show]b[showflag]b", KFMSwitch::Create, 0);
	env->AddFunction("AssertOnCUDA", "c", AssertOnCUDA::Create, 0);
}
