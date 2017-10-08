
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>

#include "CommonFunctions.h"
#include "TextOut.h"
#include "KMV.h"
#include "KFM.h"

void OnCudaError(cudaError_t err) {
#if 1 // デバッグ用（本番は取り除く）
  printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
#endif
}

int GetDeviceType(const PClip& clip)
{
  int devtypes = (clip->GetVersion() >= 5) ? clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  if (devtypes == 0) {
    return DEV_TYPE_CPU;
  }
  return devtypes;
}

#define COMBE_FLAG_STR "KRemoveCombe_Flag"

template <typename pixel_t>
void Copy(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * dst_pitch] = src[x + y * src_pitch];
    }
  }
}

class KShowStatic : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  PClip sttclip;

  int logUVx;
  int logUVy;

  void CopyFrame(PVideoFrame& src, PVideoFrame& dst)
  {
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

    Copy<pixel_t>(dstY, pitchY, srcY, pitchY, vi.width, vi.height);
    Copy<pixel_t>(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV);
    Copy<pixel_t>(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV);
  }

  void MaskFill(pixel_t* dstp, int dstPitch,
    const uint8_t* flagp, int flagPitch, int width, int height, int val)
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int coef = flagp[x + y * flagPitch];
        pixel_t& v = dstp[x + y * dstPitch];
        v = (coef * v + (128 - coef) * val + 64) >> 7;
      }
    }
  }

  void VisualizeBlock(PVideoFrame& flag, PVideoFrame& dst)
  {
    const pixel_t* flagY = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_Y));
    const pixel_t* flagU = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_U));
    const pixel_t* flagV = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int flagPitchY = flag->GetPitch(PLANAR_Y);
    int flagPitchUV = flag->GetPitch(PLANAR_U);
    int dstPitchY = dst->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int dstPitchUV = dst->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    int blue[] = { 73, 230, 111 };

    MaskFill(dstY, dstPitchY, flagY, flagPitchY, vi.width, vi.height, blue[0]);
    MaskFill(dstU, dstPitchUV, flagU, flagPitchUV, widthUV, heightUV, blue[1]);
    MaskFill(dstV, dstPitchUV, flagV, flagPitchUV, widthUV, heightUV, blue[2]);
  }

public:
  KShowStatic(PClip sttclip, PClip clip30, IScriptEnvironment2* env)
    : GenericVideoFilter(clip30)
    , sttclip(sttclip)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    PVideoFrame flag = sttclip->GetFrame(n, env);
    PVideoFrame frame30 = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    CopyFrame(frame30, dst);
    VisualizeBlock(flag, dst);

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    return new KShowStatic(
      args[0].AsClip(),       // sttclip
      args[1].AsClip(),       // clip30
      env);
  }
};

class KMergeStatic : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  PClip clip30;
  PClip sttclip;

  int logUVx;
  int logUVy;

  int nBlkX;
  int nBlkY;

  bool ContainsStaticBlock(PVideoFrame& block)
  {
    const uint8_t* blockp = reinterpret_cast<const uint8_t*>(block->GetReadPtr());

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        if (blockp[bx + by * nBlkX]) return true;
      }
    }

    return false;
  }

  void CopyFrame(PVideoFrame& src, PVideoFrame& dst)
  {
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

    Copy<pixel_t>(dstY, pitchY, srcY, pitchY, vi.width, vi.height);
    Copy<pixel_t>(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV);
    Copy<pixel_t>(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV);
  }

  void MergeStatic(pixel_t* dstp, const pixel_t* src60, const pixel_t* src30, int pitch,
    const pixel_t* flagp, int width, int height)
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int coef = flagp[x + y * pitch];
        int v30 = src30[x + y * pitch];
        int v60 = src60[x + y * pitch];
        dstp[x + y * pitch] = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
      }
    }
  }

  void MergeStatic(PVideoFrame& src60, PVideoFrame& src30, PVideoFrame& flag, PVideoFrame& dst)
  {
    const pixel_t* src60Y = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_Y));
    const pixel_t* src60U = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_U));
    const pixel_t* src60V = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_V));
    const pixel_t* src30Y = reinterpret_cast<const pixel_t*>(src30->GetReadPtr(PLANAR_Y));
    const pixel_t* src30U = reinterpret_cast<const pixel_t*>(src30->GetReadPtr(PLANAR_U));
    const pixel_t* src30V = reinterpret_cast<const pixel_t*>(src30->GetReadPtr(PLANAR_V));
    const pixel_t* flagY = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_Y));
    const pixel_t* flagU = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_U));
    const pixel_t* flagV = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src60->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src60->GetPitch(PLANAR_U) / sizeof(pixel_t);

    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    MergeStatic(dstY, src60Y, src30Y, pitchY, flagY, vi.width, vi.height);
    MergeStatic(dstU, src60U, src30U, pitchUV, flagU, widthUV, heightUV);
    MergeStatic(dstV, src60V, src30V, pitchUV, flagV, widthUV, heightUV);
  }

public:
  KMergeStatic(PClip clip60, PClip clip30, PClip sttclip, IScriptEnvironment2* env)
    : GenericVideoFilter(clip60)
    , clip30(clip30)
    , sttclip(sttclip)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    VideoInfo srcvi = clip30->GetVideoInfo();
    if (vi.num_frames != srcvi.num_frames * 2) {
      env->ThrowError("[KMergeDev] Num frames don't match");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int n30 = n >> 1;
    PVideoFrame flag = sttclip->GetFrame(n30, env);
    PVideoFrame frame60 = child->GetFrame(n, env);
    PVideoFrame frame30 = clip30->GetFrame(n30, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    CopyFrame(frame60, dst);
    MergeStatic(frame60, frame30, flag, dst);

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    return new KMergeStatic(
      args[0].AsClip(),       // clip60
      args[1].AsClip(),       // clip30
      args[2].AsClip(),       // sttclip
      env);
  }
};

enum {
	MOVE = 1,
	SHIMA = 2,
	LSHIMA = 4,

	BORDER = 4,
};

// srefはbase-1ライン
template <typename pixel_t>
static void CompareFields(pixel_t* dst, int dstPitch,
	const pixel_t* base, const pixel_t* sref, const pixel_t* mref,
	int width, int height, int pitch, int threshM, int threshS, int threshLS)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			pixel_t flag = 0;

			// 縞判定
			pixel_t a = base[x + (y - 1) * pitch];
			pixel_t b = sref[x + y * pitch];
			pixel_t c = base[x + y * pitch];
			pixel_t d = sref[x + (y + 1) * pitch];
			pixel_t e = base[x + (y + 1) * pitch];

			int t = (a + 4 * c + e - 3 * (b + d));
			if (t > threshS) flag |= SHIMA;
			if (t > threshLS) flag |= LSHIMA;

			// 動き判定
			pixel_t f = mref[x + y * pitch];
			if (std::abs(f - c) > threshM) flag |= MOVE;

			// フラグ格納
			dst[x + y * dstPitch] = flag;
		}
	}
}

static void MergeUVFlags(PVideoFrame& fflag, int width, int height, int logUVx, int logUVy)
{
	uint8_t* fY = fflag->GetWritePtr(PLANAR_Y);
	const uint8_t* fU = fflag->GetWritePtr(PLANAR_U);
	const uint8_t* fV = fflag->GetWritePtr(PLANAR_V);
	int pitchY = fflag->GetPitch(PLANAR_Y);
	int pitchUV = fflag->GetPitch(PLANAR_U);

	for (int y = BORDER; y < height - BORDER; ++y) {
		for (int x = 0; x < width; ++x) {
			int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
			int flagUV = fU[offUV] | fV[offUV];
			fY[x + y * pitchY] |= (flagUV << 4);
		}
	}
}

struct FMCount {
	int move, shima, lshima;
};

static FMCount CountFlags(const uint8_t* flagp, int width, int height, int pitch)
{
	FMCount cnt = { 0 };
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int flag = flagp[x + y * pitch];
			if (flag & MOVE) cnt.move++;
			if (flag & SHIMA) cnt.shima++;
			if (flag & LSHIMA) cnt.lshima++;
		}
	}
  return cnt;
}

class KFMFrameDev : public GenericVideoFilter
{
	typedef uint8_t pixel_t;

  VideoInfo blockvi;

	int threshMY;
	int threshSY;
	int threshLSY;
	int threshMC;
	int threshSC;
	int threshLSC;

	int logUVx;
	int logUVy;

	void VisualizeFlags(PVideoFrame& dst, PVideoFrame& fflag)
	{
		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		// 判定結果を表示
		int black[] = { 0, 128, 128 };
		int blue[] = { 73, 230, 111 };
		int gray[] = { 140, 128, 128 };
		int purple[] = { 197, 160, 122 };

		const pixel_t* fflagp = reinterpret_cast<const pixel_t*>(fflag->GetReadPtr(PLANAR_Y));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int flagPitch = fflag->GetPitch(PLANAR_Y);
		int dstPitchY = dst->GetPitch(PLANAR_Y);
		int dstPitchUV = dst->GetPitch(PLANAR_U);

		// 黒で初期化しておく
		for (int y = 0; y < vi.height; ++y) {
			for (int x = 0; x < vi.width; ++x) {
				int offY = x + y * dstPitchY;
				int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
				dstY[offY] = black[0];
				dstU[offUV] = black[1];
				dstV[offUV] = black[2];
			}
		}

		// 色を付ける
		for (int y = BORDER; y < vi.height - BORDER; ++y) {
			for (int x = 0; x < vi.width; ++x) {
				int flag = fflagp[x + y * flagPitch];
				flag |= (flag >> 4);

				int* color = nullptr;
				if ((flag & MOVE) && (flag & SHIMA)) {
					color = purple;
				}
				else if (flag & MOVE) {
					color = blue;
				}
				else if (flag & SHIMA) {
					color = gray;
				}

				if (color) {
					int offY = x + y * dstPitchY;
					int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
					dstY[offY] = color[0];
					dstU[offUV] = color[1];
					dstV[offUV] = color[2];
				}
			}
		}
	}

public:
	KFMFrameDev(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
		: GenericVideoFilter(clip)
		, threshMY(threshMY)
		, threshSY(threshSY * 6)
		, threshLSY(threshSY * 6 * 3)
		, threshMC(threshMC)
		, threshSC(threshSC * 6)
		, threshLSC(threshSC * 6 * 3)
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , blockvi(vi)
	{
    if (vi.Is420()) blockvi.pixel_type = VideoInfo::CS_YV12;
    else if (vi.Is422()) blockvi.pixel_type = VideoInfo::CS_YV16;
    else if (vi.Is444()) blockvi.pixel_type = VideoInfo::CS_YV24;
    blockvi.width = vi.width;
    blockvi.height = vi.height;
  }

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame f0 = child->GetFrame(n, env);
		PVideoFrame f1 = child->GetFrame(n + 1, env);
		PVideoFrame fflag = env->NewVideoFrame(blockvi);

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		// 各プレーンを判定
		for (int pi = 0; pi < 3; ++pi) {
			int p = planes[pi];

			const pixel_t* f0p = reinterpret_cast<const pixel_t*>(f0->GetReadPtr(p));
			const pixel_t* f1p = reinterpret_cast<const pixel_t*>(f1->GetReadPtr(p));
			pixel_t* fflagp = reinterpret_cast<pixel_t*>(fflag->GetWritePtr(p));
			int pitch = f0->GetPitch(p);
			int dstPitch = fflag->GetPitch(p);

			// 計算が面倒なので上下4ピクセルは除外
			int width = vi.width;
			int height = vi.height;
			int border = BORDER;
			if (pi > 0) {
				width >>= logUVx;
				height >>= logUVy;
				border >>= logUVy;
			}

			int threshM = (pi == 0) ? threshMY : threshMC;
			int threshS = (pi == 0) ? threshSY : threshSC;
			int threshLS = (pi == 0) ? threshLSY : threshLSC;

			// top
			CompareFields(
				fflagp + dstPitch * border, dstPitch * 2,
				f0p + pitch * border,
				f0p + pitch * (border - 1),
				f1p + pitch * border,
				width, (height - border * 2) / 2, pitch * 2,
				threshM, threshS, threshLS);

			// bottom
			CompareFields(
				fflagp + dstPitch * (border + 1), dstPitch * 2,
				f0p + pitch * (border + 1),
				f1p + pitch * border,
				f1p + pitch * (border + 1),
				width, (height - border * 2) / 2, pitch * 2,
				threshM, threshS, threshLS);
		}

		// UV判定結果をYにマージ
		MergeUVFlags(fflag, vi.width, vi.height, logUVx, logUVy);

		// 判定結果を表示
		PVideoFrame dst = env->NewVideoFrame(vi);
		VisualizeFlags(dst, fflag);

		return dst;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KFMFrameDev(
			args[0].AsClip(),       // clip
			args[1].AsInt(10),       // threshMY
			args[2].AsInt(10),       // threshSY
			args[3].AsInt(10),       // threshMC
			args[4].AsInt(10),       // threshSC
			env
			);
	}
};

class KFMFrameAnalyze : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  VideoInfo srcvi;

  int threshMY;
  int threshSY;
  int threshLSY;
  int threshMC;
  int threshSC;
  int threshLSC;

  int logUVx;
  int logUVy;

public:
  KFMFrameAnalyze(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
		: GenericVideoFilter(clip)
    , threshMY(threshMY)
    , threshSY(threshSY * 6)
    , threshLSY(threshSY * 6 * 3)
    , threshMC(threshMC)
    , threshSC(threshSC * 6)
    , threshLSC(threshSC * 6 * 3)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{
    srcvi = vi;

		int out_bytes = sizeof(FMCount) * 2;
		vi.pixel_type = VideoInfo::CS_BGR32;
		vi.width = 16;
		vi.height = nblocks(out_bytes, vi.width * 4);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		int parity = child->GetParity(n);
		PVideoFrame f0 = child->GetFrame(n, env);
		PVideoFrame f1 = child->GetFrame(n + 1, env);

    VideoInfo flagvi = VideoInfo();
    if (srcvi.Is420()) flagvi.pixel_type = VideoInfo::CS_YV12;
    else if (srcvi.Is422()) flagvi.pixel_type = VideoInfo::CS_YV16;
    else if (srcvi.Is444()) flagvi.pixel_type = VideoInfo::CS_YV24;
    flagvi.width = srcvi.width;
    flagvi.height = srcvi.height;
    PVideoFrame fflag = env->NewVideoFrame(flagvi);

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		// 各プレーンを判定
		for (int pi = 0; pi < 3; ++pi) {
      int p = planes[pi];

      const pixel_t* f0p = reinterpret_cast<const pixel_t*>(f0->GetReadPtr(p));
      const pixel_t* f1p = reinterpret_cast<const pixel_t*>(f1->GetReadPtr(p));
      pixel_t* fflagp = reinterpret_cast<pixel_t*>(fflag->GetWritePtr(p));
      int pitch = f0->GetPitch(p);
      int dstPitch = fflag->GetPitch(p);

      // 計算が面倒なので上下4ピクセルは除外
      int width = srcvi.width;
      int height = srcvi.height;
      int border = BORDER;
      if (pi > 0) {
        width >>= logUVx;
        height >>= logUVy;
        border >>= logUVy;
      }

      int threshM = (pi == 0) ? threshMY : threshMC;
      int threshS = (pi == 0) ? threshSY : threshSC;
      int threshLS = (pi == 0) ? threshLSY : threshLSC;

      // top
      CompareFields(
        fflagp + dstPitch * border, dstPitch * 2,
        f0p + pitch * border,
        f0p + pitch * (border - 1),
        f1p + pitch * border,
        width, (height - border * 2) / 2, pitch * 2,
        threshM, threshS, threshLS);

      // bottom
      CompareFields(
        fflagp + dstPitch * (border + 1), dstPitch * 2,
        f0p + pitch * (border + 1),
        f1p + pitch * border,
        f1p + pitch * (border + 1),
        width, (height - border * 2) / 2, pitch * 2,
        threshM, threshS, threshLS);
		}

    // UV判定結果をYにマージ
    MergeUVFlags(fflag, srcvi.width, srcvi.height, logUVx, logUVy);

		PVideoFrame dst = env->NewVideoFrame(vi);
    uint8_t* dstp = dst->GetWritePtr();

		// CountFlags
    const pixel_t* fflagp = reinterpret_cast<const pixel_t*>(fflag->GetReadPtr(PLANAR_Y));
    int flagPitch = fflag->GetPitch(PLANAR_Y);
    FMCount fmcnt[2];
    fmcnt[0] = CountFlags(fflagp + BORDER * flagPitch, srcvi.width, (srcvi.height - BORDER * 2) / 2, flagPitch * 2);
    fmcnt[1] = CountFlags(fflagp + (BORDER + 1) * flagPitch, srcvi.width, (srcvi.height - BORDER * 2) / 2, flagPitch * 2);
    if (!parity) {
      std::swap(fmcnt[0], fmcnt[1]);
    }
    memcpy(dstp, fmcnt, sizeof(FMCount) * 2);

		return dst;
	}

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyze(
      args[0].AsClip(),       // clip
      args[1].AsInt(15),       // threshMY
      args[2].AsInt(7),       // threshSY
      args[3].AsInt(20),       // threshMC
      args[4].AsInt(8),       // threshSC
      env
    );
  }
};

struct FMData {
	// 小さい縞の量
	float fieldv[14];
	float fieldbase[14];
	// 大きい縞の量
	float fieldlv[14];
	float fieldlbase[14];
	// 動き量
	float move[14];
	// 動きから見た分離度
	float splitv[14];
	// 動きから見た3フィールド結合度
	float mergev[14];
};

struct PulldownPatternField {
	bool split; // 次のフィールドとは別フレーム
	bool merge; // 3フィールドの最初のフィールド
};

void CalcBaseline(const float* data, float* baseline, int N)
{
  baseline[0] = data[0];
  float ra = FLT_MAX, ry;
  for (int pos = 0; pos < N - 1;) {
    float mina = FLT_MAX;
    int ni;
    for (int i = 1; pos + i < N; ++i) {
      float a = (data[pos + i] - data[pos]) / i;
      if (a < mina) {
        mina = a; ni = i;
      }
    }
    bool ok = true;
    for (int i = 0; i < N; ++i) {
      if (data[i] < data[pos] + mina * (i - pos)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      if (std::abs(mina) < std::abs(ra)) {
        ra = mina;
        ry = data[pos] + mina * (-pos);
      }
    }
    pos = pos + ni;
  }
  for (int i = 0; i < N; ++i) {
    baseline[i] = ry + ra * i;
  }
}

struct PulldownPattern {
	PulldownPatternField fields[10 * 4];

	PulldownPattern(int nf0, int nf1, int nf2, int nf3)
		: fields()
	{
		if (nf0 + nf1 + nf2 + nf3 != 10) {
			printf("Error: sum of nfields must be 10.\n");
		}
		int nfields[] = { nf0, nf1, nf2, nf3 };
		for (int c = 0, fstart = 0; c < 4; ++c) {
			for (int i = 0; i < 4; ++i) {
				int nf = nfields[i];
				for (int f = 0; f < nf - 2; ++f) {
					fields[fstart + f].merge = true;
				}
				fields[fstart + nf - 1].split = true;
				fstart += nf;
			}
		}
	}

	const PulldownPatternField* GetPattern(int n) const {
		return &fields[10 + n - 2];
	}
};

float SplitScore(const PulldownPatternField* pattern, const float* fv, const float* base) {
	int nsplit = 0, nnsplit = 0;
	float sumsplit = 0, sumnsplit = 0;

	for (int i = 0; i < 14; ++i) {
		if (pattern[i].split) {
			nsplit++;
			sumsplit += fv[i] - (base ? base[i] : 0);
		}
		else {
			nnsplit++;
			sumnsplit += fv[i] - (base ? base[i] : 0);
		}
	}
	
  float splitcoef = sumsplit / nsplit;
  float nsplitcoef = sumnsplit / nnsplit;
  if (nsplitcoef == 0 && splitcoef == 0) {
    return 0;
  }
  return splitcoef / (nsplitcoef + 0.1f * splitcoef);
}

float SplitCost(const PulldownPatternField* pattern, const float* fv) {
  int nsplit = 0, nnsplit = 0;
  float sumsplit = 0, sumnsplit = 0;

  for (int i = 0; i < 14; ++i) {
    if (pattern[i].split) {
      nsplit++;
      sumsplit += fv[i];
    }
    else {
      nnsplit++;
      sumnsplit += fv[i];
    }
  }

  float splitcoef = sumsplit / nsplit;
  float nsplitcoef = sumnsplit / nnsplit;
  if (nsplitcoef == 0 && splitcoef == 0) {
    return 0;
  }
  return nsplitcoef / (splitcoef + 0.1f * nsplitcoef);
}

float MergeScore(const PulldownPatternField* pattern, const float* mergev) {
	int nmerge = 0, nnmerge = 0;
	float summerge = 0, sumnmerge = 0;

	for (int i = 0; i < 14; ++i) {
		if (pattern[i].merge) {
			nmerge++;
			summerge += mergev[i];
		}
		else {
			nnmerge++;
			sumnmerge += mergev[i];
		}
	}

  float splitcoef = summerge / nmerge;
  float nsplitcoef = sumnmerge / nnmerge;
  if (nsplitcoef == 0 && splitcoef == 0) {
    return 0;
  }
  return splitcoef / (nsplitcoef + 0.1f * splitcoef);
}

struct Frame24InfoV2 {
  int cycleIndex;
  int frameIndex; // サイクル内のフレーム番号
  int fieldStartIndex; // ソースフィールド開始番号
  int numFields; // ソースフィールド数
};

class PulldownPatterns
{
	PulldownPattern p2323, p2233, p2224;
	const PulldownPatternField* allpatterns[27];
public:
	PulldownPatterns()
		: p2323(2, 3, 2, 3)
		, p2233(2, 2, 3, 3)
		, p2224(2, 2, 2, 4)
	{
		const PulldownPattern* patterns[] = { &p2323, &p2233, &p2224 };

		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < 9; ++i) {
				allpatterns[p * 9 + i] = patterns[p]->GetPattern(i);
			}
		}
	}

	const PulldownPatternField* GetPattern(int patternIndex) const {
		return allpatterns[patternIndex];
	}

  // パターンと24fpsのフレーム番号からフレーム情報を取得
  Frame24InfoV2 GetFrame24(int patternIndex, int n24) const {
    Frame24InfoV2 info;
    info.cycleIndex = n24 / 4;
    info.frameIndex = n24 % 4;

    const PulldownPatternField* ptn = allpatterns[patternIndex];
    int fldstart = 0;
    int nframes = 0;
    for (int i = 0; i < 14; ++i) {
      if (ptn[i].split) {
        if (fldstart >= 1) {
          if (nframes++ == info.frameIndex) {
            int nextfldstart = i + 1;
            info.fieldStartIndex = fldstart - 2;
            info.numFields = nextfldstart - fldstart;
            return info;
          }
        }
        fldstart = i + 1;
      }
    }

    throw "Error !!!";
	}

  // パターンと60fpsのフレーム番号からフレーム情報を取得
  // frameIndex < 0 or frameIndex >= 4の場合、
  // fieldStartIndexとnumFieldsは正しくない可能性があるので注意
  Frame24InfoV2 GetFrame60(int patternIndex, int n60) const {
    Frame24InfoV2 info;
    info.cycleIndex = n60 / 10;
    
    const PulldownPatternField* ptn = allpatterns[patternIndex];
    int fldstart = 0;
    int nframes = -1;
    int findex = n60 % 10;
    for (int i = 0; i < 14; ++i) {
      if (ptn[i].split) {
        if (fldstart >= 1) {
          ++nframes;
        }
        int nextfldstart = i + 1;
        if (findex < nextfldstart - 2) {
          info.frameIndex = nframes;
          info.fieldStartIndex = fldstart - 2;
          info.numFields = nextfldstart - fldstart;
          return info;
        }
        fldstart = i + 1;
      }
    }
    
    info.frameIndex = ++nframes;
    info.fieldStartIndex = fldstart - 2;
    info.numFields = 14 - fldstart;
    return info;
  }

	std::pair<int, float> Matching(const FMData* data, int width, int height) const
	{
		const PulldownPattern* patterns[] = { &p2323, &p2233, &p2224 };

		// 1ピクセル当たりの縞の数
		float shimaratio = std::accumulate(data->fieldv, data->fieldv + 14, 0.0f) / (14.0f * width * height);
		float lshimaratio = std::accumulate(data->fieldlv, data->fieldlv + 14, 0.0f) / (14.0f * width * height);
		float moveratio = std::accumulate(data->move, data->move + 14, 0.0f) / (14.0f * width * height);

		std::vector<float> shima(9 * 3);
		std::vector<float> shimabase(9 * 3);
		std::vector<float> lshima(9 * 3);
		std::vector<float> lshimabase(9 * 3);
		std::vector<float> split(9 * 3);
    std::vector<float> merge(9 * 3);

    std::vector<float> mergecost(9 * 3);
    std::vector<float> shimacost(9 * 3);
    std::vector<float> lshimacost(9 * 3);

		// 各スコアを計算
		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < 9; ++i) {

				auto pattern = patterns[p]->GetPattern(i);
				shima[p * 9 + i] = SplitScore(pattern, data->fieldv, 0);
				shimabase[p * 9 + i] = SplitScore(pattern, data->fieldv, data->fieldbase);
				lshima[p * 9 + i] = SplitScore(pattern, data->fieldlv, 0);
				lshimabase[p * 9 + i] = SplitScore(pattern, data->fieldlv, data->fieldlbase);
				split[p * 9 + i] = SplitScore(pattern, data->splitv, 0);
        merge[p * 9 + i] = MergeScore(pattern, data->mergev);

        shimacost[p * 9 + i] = SplitCost(pattern, data->fieldv);
        lshimacost[p * 9 + i] = SplitCost(pattern, data->fieldlv);
        mergecost[p * 9 + i] = MergeScore(pattern, data->move);
			}
		}

    auto makeRet = [&](int n) {
      float cost = shimacost[n] + lshimacost[n] + mergecost[n];
      return std::pair<int, float>(n, cost);
    };

		auto it = std::max_element(shima.begin(), shima.end());
    if (moveratio >= 0.002f) { // TODO:
      if (*it >= 2.0f) {
        // 高い確度で特定
        return makeRet(int(it - shima.begin()));
      }
      it = std::max_element(lshima.begin(), lshima.end());
      if (*it >= 4.0f) {
        // 高い確度で特定
        return makeRet(int(it - lshima.begin()));
      }
    }

		// ほぼ止まってる or ノイズ成分or文字による誤爆が多い or テロップなどが邪魔してる //

    // マージで判定できるか
    int nummerge = (int)std::count_if(data->mergev, data->mergev + 14, [](float f) { return f > 1.0f; });
    if (nummerge == 3 || nummerge == 4) {
      // 十分な数のマージがある
      it = std::max_element(merge.begin(), merge.end());
      if (*it >= 5.0f) {
        return makeRet(int(it - merge.begin()));
      }
    }

    // テロップ・文字などの固定成分を抜いて比較
    std::vector<float> scores(9 * 3);
    for (int i = 0; i < (int)scores.size(); ++i) {
      scores[i] = split[i];

      if (shimaratio > 0.002f) {
        // 小さい縞が十分あるときは小さい縞だけで判定する
        scores[i] += shimabase[i];
      }
      else if (moveratio < 0.002f) {
        // 小さい縞は動きが多いときだけにする（動きが小さいと文字とかで誤爆しやすいので）
        scores[i] += lshimabase[i];
      }
      else {
        scores[i] += shimabase[i] + lshimabase[i];
      }
    }
    it = std::max_element(scores.begin(), scores.end());

    return makeRet(int(it - scores.begin()));
	}
};

class KFMCycleAnalyze : public GenericVideoFilter
{
	PClip source;
  VideoInfo srcvi;
	PulldownPatterns patterns;
public:
  KFMCycleAnalyze(PClip fmframe, PClip source, IScriptEnvironment* env)
		: GenericVideoFilter(fmframe)
		, source(source)
    , srcvi(source->GetVideoInfo())
	{
    int out_bytes = sizeof(std::pair<int, float>);
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);
  }

	PVideoFrame __stdcall GetFrame(int cycle, IScriptEnvironment* env)
	{
		FMCount fmcnt[18];
		for (int i = -2; i <= 6; ++i) {
			PVideoFrame frame = child->GetFrame(cycle * 5 + i, env);
			memcpy(fmcnt + (i + 2) * 2, frame->GetReadPtr(), sizeof(fmcnt[0]) * 2);
		}

		FMData data = { 0 };
		int fieldbase = INT_MAX;
		int fieldlbase = INT_MAX;
    for (int i = 0; i < 14; ++i) {
			data.fieldv[i] = (float)fmcnt[i + 2].shima;
			data.fieldlv[i] = (float)fmcnt[i + 2].lshima;
			data.move[i] = (float)fmcnt[i + 2].move;
      data.splitv[i] = (float)std::min(fmcnt[i + 1].move, fmcnt[i + 2].move);
			fieldbase = std::min(fieldbase, fmcnt[i + 2].shima);
			fieldlbase = std::min(fieldlbase, fmcnt[i + 2].lshima);

      if (fmcnt[i + 1].move > fmcnt[i + 2].move && fmcnt[i + 3].move > fmcnt[i + 2].move) {
        float sum = (float)(fmcnt[i + 1].move + fmcnt[i + 3].move);
        data.mergev[i] = std::max(1.0f, sum / (fmcnt[i + 2].move * 2.0f + 0.1f * sum)) - 1.0f;
      }
      else {
        data.mergev[i] = 0;
      }
		}
    CalcBaseline(data.fieldv, data.fieldbase, 14);
    CalcBaseline(data.fieldlv, data.fieldlbase, 14);

		auto result = patterns.Matching(&data, srcvi.width, srcvi.height);

    PVideoFrame dst = env->NewVideoFrame(vi);
    uint8_t* dstp = dst->GetWritePtr();
    memcpy(dstp, &result, sizeof(result));

    return dst;
	}

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMCycleAnalyze(
      args[0].AsClip(),       // fmframe
      args[1].AsClip(),       // source
      env
    );
  }
};

class KTelecineCoreBase
{
public:
	virtual ~KTelecineCoreBase() { }
	virtual void CreateWeaveFrame2F(const PVideoFrame& srct, const PVideoFrame& srcb, const PVideoFrame& dst) = 0;
};

template <typename pixel_t>
class KTelecineCore : public KTelecineCoreBase
{
	const VideoInfo& vi;
	int logUVx;
	int logUVy;
public:
	KTelecineCore(const VideoInfo& vi)
		: vi(vi)
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{ }

	void CreateWeaveFrame2F(const PVideoFrame& srct, const PVideoFrame& srcb, const PVideoFrame& dst)
	{
		const pixel_t* srctY = reinterpret_cast<const pixel_t*>(srct->GetReadPtr(PLANAR_Y));
		const pixel_t* srctU = reinterpret_cast<const pixel_t*>(srct->GetReadPtr(PLANAR_U));
		const pixel_t* srctV = reinterpret_cast<const pixel_t*>(srct->GetReadPtr(PLANAR_V));
		const pixel_t* srcbY = reinterpret_cast<const pixel_t*>(srcb->GetReadPtr(PLANAR_Y));
		const pixel_t* srcbU = reinterpret_cast<const pixel_t*>(srcb->GetReadPtr(PLANAR_U));
		const pixel_t* srcbV = reinterpret_cast<const pixel_t*>(srcb->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = srct->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int pitchUV = srct->GetPitch(PLANAR_U) / sizeof(pixel_t);
		int widthUV = vi.width >> logUVx;
		int heightUV = vi.height >> logUVy;

		// copy top
		Copy<pixel_t>(dstY, pitchY * 2, srctY, pitchY * 2, vi.width, vi.height / 2);
		Copy<pixel_t>(dstU, pitchUV * 2, srctU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV, pitchUV * 2, srctV, pitchUV * 2, widthUV, heightUV / 2);

		// copy bottom
		Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcbY + pitchY, pitchY * 2, vi.width, vi.height / 2);
		Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcbU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcbV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
	}
};

class KTelecine : public GenericVideoFilter
{
	PClip fmclip;
  bool show;

  PulldownPatterns patterns;

	std::unique_ptr<KTelecineCoreBase> core;

	KTelecineCoreBase* CreateCore(IScriptEnvironment* env)
	{
		if (vi.ComponentSize() == 1) {
			return new KTelecineCore<uint8_t>(vi);
		}
		else {
			return new KTelecineCore<uint16_t>(vi);
		}
	}

	PVideoFrame CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, int parity, IScriptEnvironment* env)
	{
		// fstartは0or1にする
		if (fstart < 0 || fstart >= 2) {
			n += fstart / 2;
			fstart &= 1;
		}

		assert(fstart == 0 || fstart == 1);
		assert(fnum == 2 || fnum == 3 || fnum == 4);

		if (fstart == 0 && fnum == 2) {
			return clip->GetFrame(n, env);
		}
		else {
			PVideoFrame cur = clip->GetFrame(n, env);
			PVideoFrame nxt = clip->GetFrame(n + 1, env);
			PVideoFrame dst = env->NewVideoFrame(vi);
			if (parity) {
				core->CreateWeaveFrame2F(nxt, cur, dst);
			}
			else {
				core->CreateWeaveFrame2F(cur, nxt, dst);
			}
			return dst;
		}
	}

	void DrawInfo(PVideoFrame& dst, const std::pair<int, float>* fmframe, int fnum, IScriptEnvironment* env) {
		env->MakeWritable(&dst);

		char buf[100]; sprintf(buf, "KFM: %d (%.1f) - %d", fmframe->first, fmframe->second, fnum);
		DrawText(dst, true, 0, 0, buf);
	}

public:
	KTelecine(PClip child, PClip fmclip, bool show, IScriptEnvironment* env)
		: GenericVideoFilter(child)
    , fmclip(fmclip)
    , show(show)
  {
		core = std::unique_ptr<KTelecineCoreBase>(CreateCore(env));

		// フレームレート
		vi.MulDivFPS(4, 5);
		vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		int cycleIndex = n / 4;
		int parity = child->GetParity(cycleIndex * 5);
		PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
		const std::pair<int, float>* fmframe = (std::pair<int, float>*)fm->GetReadPtr();
		int pattern = fmframe->first;
    Frame24InfoV2 frameInfo = patterns.GetFrame24(pattern, n);

    int fstart = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
    PVideoFrame out = CreateWeaveFrame(child, 0, fstart, frameInfo.numFields, parity, env);

    if (show) {
      DrawInfo(out, fmframe, frameInfo.numFields, env);
    }

		return out;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KTelecine(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // fmclip
      args[2].AsBool(false),  // show
			env
			);
	}
};

class KRemoveCombe : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  VideoInfo padvi;
  VideoInfo blockvi;
  int logUVx;
  int logUVy;
  int nBlkX, nBlkY;

  float thsmooth;
  float smooth;
  float thcombe;
  float ratio1;
  float ratio2;
  bool outcombe;
  bool show;

  void PadFrame(PVideoFrame& dst)
  {
    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int pi = 0; pi < 3; ++pi) {
      int p = planes[pi];

      pixel_t* dstptr = reinterpret_cast<pixel_t*>(dst->GetWritePtr(p));
      int pitch = dst->GetPitch(p) / sizeof(pixel_t);

      int width = vi.width;
      int height = vi.height;
      int vpad = VPAD;
      if (pi > 0) {
        width >>= logUVx;
        height >>= logUVy;
        vpad >>= logUVy;
      }

      for (int y = 0; y < vpad; ++y) {
        for (int x = 0; x < width; ++x) {
          dstptr[x + (-y - 1) * pitch] = dstptr[x + (y)* pitch];
          dstptr[x + (height + y) * pitch] = dstptr[x + (height - y - 1)* pitch];
        }
      }
    }
  }

  void CopyFrame(PVideoFrame& src, PVideoFrame& dst)
  {
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

    Copy<pixel_t>(dstY, pitchY, srcY, pitchY, vi.width, vi.height);
    Copy<pixel_t>(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV);
    Copy<pixel_t>(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV);
  }

  static int CalcCombe(int a, int b, int c, int d, int e) {
    return (a + 4 * c + e - 3 * (b + d));
  }

  void CompareFields(pixel_t* dst, int dstPitch,
    const pixel_t* src,
    int width, int height, int pitch, int thresh)
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        pixel_t flag = 0;

        // 縞判定
        int t = CalcCombe(
          src[x + (y - 2) * pitch],
          src[x + (y - 1) * pitch],
          src[x + (y + 0) * pitch],
          src[x + (y + 1) * pitch],
          src[x + (y + 2) * pitch]);

        if (t > thresh) flag = 1;

        // フラグ格納
        dst[x + y * dstPitch] = flag;
      }
    }
  }

  void FindCombe3(PVideoFrame& src, PVideoFrame& flag, int thresh)
  {
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(flag->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    CompareFields(dstY, pitchY, srcY, vi.width, vi.height, pitchY, thresh);
    CompareFields(dstU, pitchUV, srcU, widthUV, heightUV, pitchUV, thresh);
    CompareFields(dstV, pitchUV, srcV, widthUV, heightUV, pitchUV, thresh);
  }

  void ExtendFlag(pixel_t* dst , const pixel_t* src, int width, int height, int pitch)
  {
    for (int y = 1; y < height - 1; ++y) {
      for (int x = 0; x < width; ++x) {
        dst[x + y * pitch] = src[x + (y - 1) * pitch] | src[x + y * pitch] | src[x + (y + 1) * pitch];
      }
    }
  }

  void ExtendFlag(PVideoFrame& src, PVideoFrame& dst)
  {
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

    ExtendFlag(dstY, srcY, vi.width, vi.height, pitchY);
    ExtendFlag(dstU, srcU, widthUV, heightUV, pitchUV);
    ExtendFlag(dstV, srcV, widthUV, heightUV, pitchUV);
  }

  void MergeUVFlags(PVideoFrame& fflag)
  {
    uint8_t* fY = fflag->GetWritePtr(PLANAR_Y);
    uint8_t* fU = fflag->GetWritePtr(PLANAR_U);
    uint8_t* fV = fflag->GetWritePtr(PLANAR_V);
    int pitchY = fflag->GetPitch(PLANAR_Y);
    int pitchUV = fflag->GetPitch(PLANAR_U);

    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
        fY[x + y * pitchY] |= fU[offUV] | fV[offUV];
      }
    }
  }

  void ApplyUVFlags(PVideoFrame& fflag)
  {
    uint8_t* fY = fflag->GetWritePtr(PLANAR_Y);
    uint8_t* fU = fflag->GetWritePtr(PLANAR_U);
    uint8_t* fV = fflag->GetWritePtr(PLANAR_V);
    int pitchY = fflag->GetPitch(PLANAR_Y);
    int pitchUV = fflag->GetPitch(PLANAR_U);

    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int offUV = (x >> logUVx) + (y >> logUVy) * pitchUV;
        uint8_t s = fY[x + y * pitchY];
        fU[offUV] = s;
      }
    }
  }

  static int BinomialMerge(int a, int b, int c, int d, int e, int thresh)
  {
    int minv = std::min(a, std::min(b, std::min(c, std::min(d, e))));
    int maxv = std::max(a, std::max(b, std::max(c, std::max(d, e))));
    if (maxv - minv < thresh) {
      return (b + 2 * c + d + 2) >> 2;
    }
    return c;
  }

  void RemoveCombe(pixel_t* dst, const pixel_t* src, const pixel_t* flag, int width, int height, int pitch, int thresh)
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (flag[x + y * pitch]) {
          dst[x + y * pitch] = BinomialMerge(
            src[x + (y - 2) * pitch],
            src[x + (y - 1) * pitch],
            src[x + y * pitch],
            src[x + (y + 1) * pitch],
            src[x + (y + 2) * pitch],
            thresh);
        }
        else {
          dst[x + y * pitch] = src[x + y * pitch];
        }
      }
    }
  }

  void RemoveCombe(PVideoFrame& src, PVideoFrame& flag, PVideoFrame& dst, int thresh)
  {
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    const pixel_t* flagY = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_Y));
    const pixel_t* flagUV = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_U));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    RemoveCombe(dstY, srcY, flagY, vi.width, vi.height, pitchY, thresh);
    RemoveCombe(dstU, srcU, flagUV, widthUV, heightUV, pitchUV, thresh);
    RemoveCombe(dstV, srcV, flagUV, widthUV, heightUV, pitchUV, thresh);
  }

  void VisualizeFlags(PVideoFrame& dst, PVideoFrame& fflag)
  {
    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    // 判定結果を表示
    int black[] = { 0, 128, 128 };
    int blue[] = { 73, 230, 111 };
    int gray[] = { 140, 128, 128 };
    int purple[] = { 197, 160, 122 };

    const pixel_t* fflagp = reinterpret_cast<const pixel_t*>(fflag->GetReadPtr(PLANAR_Y));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int flagPitch = fflag->GetPitch(PLANAR_Y);
    int dstPitchY = dst->GetPitch(PLANAR_Y);
    int dstPitchUV = dst->GetPitch(PLANAR_U);

    // 色を付ける
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int flag = fflagp[x + y * flagPitch];

        int* color = nullptr;
        if (flag) {
          color = blue;
        }

        if (color) {
          int offY = x + y * dstPitchY;
          int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
          dstY[offY] = color[0];
          dstU[offUV] = color[1];
          dstV[offUV] = color[2];
        }
      }
    }
  }

  void CountFlags(PVideoFrame& block, PVideoFrame& flag, int ratio1, int ratio2)
  {
    const pixel_t* srcp = reinterpret_cast<const pixel_t*>(flag->GetReadPtr(PLANAR_Y));
    uint8_t* flagp = reinterpret_cast<uint8_t*>(block->GetWritePtr());

    int pitch = flag->GetPitch(PLANAR_Y) / sizeof(pixel_t);

    for (int by = 0; by < nBlkY - 1; ++by) {
      for (int bx = 0; bx < nBlkX - 1; ++bx) {
        int yStart = by * OVERLAP;
        int yEnd = yStart + BLOCK_SIZE;
        int xStart = bx * OVERLAP;
        int xEnd = xStart + BLOCK_SIZE;

        int sum = 0;
        for (int y = yStart; y < yEnd; ++y) {
          for (int x = xStart; x < xEnd; x += 4) {
            // 横4ピクセルは1個とみなす
            int b = 0;
            for (int i = 0; i < 4; ++i) {
              b |= srcp[x + i + y * pitch];
            }
            sum += b ? 1 : 0;
          }
        }

        uint8_t flag = (sum > ratio1) | ((sum > ratio2) << 1);
        flagp[(bx + 1) + (by + 1) * nBlkX] = flag;
      }
    }
  }

  void CleanBlocks(PVideoFrame& block)
  {
    // 周囲にデカイ縞がなければOKとみなす
    uint8_t* flagp = reinterpret_cast<uint8_t*>(block->GetWritePtr());

    for (int by = 1; by < nBlkY; ++by) {
      for (int bx = 1; bx < nBlkX; ++bx) {
        if (flagp[bx + by * nBlkX] == 1) {

          int yStart = std::max(by - 2, 1);
          int yEnd = std::min(by + 2, nBlkY - 1);
          int xStart = std::max(bx - 2, 1);
          int xEnd = std::min(bx + 2, nBlkX - 1);

          bool isOK = true;
          for (int y = yStart; y <= yEnd; ++y) {
            for (int x = xStart; x <= xEnd; ++x) {
              if (flagp[x + y * nBlkX] & 2) {
                // デカイ縞
                isOK = false;
              }
            }
          }

          if (isOK) {
            flagp[bx + by * nBlkX] = 0;
          }
        }
      }
    }
  }

  void ExtendBlocks(PVideoFrame& flag)
  {
    uint8_t* flagp = reinterpret_cast<uint8_t*>(flag->GetWritePtr());

    // 書き込んでいないところをゼロにする
    for (int bx = 0; bx < nBlkX; ++bx) {
      flagp[bx] = 0;
    }
    for (int by = 1; by < nBlkY; ++by) {
      flagp[by * nBlkX] = 0;
    }

    // 水平方向
    for (int by = 1; by < nBlkY; ++by) {
      for (int bx = 1; bx < nBlkX; ++bx) {
        flagp[bx - 1 + by * nBlkX] |= flagp[bx + by * nBlkX];
      }
    }
    // 垂直方向
    for (int by = 1; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        flagp[bx + (by - 1) * nBlkX] |= flagp[bx + by * nBlkX];
      }
    }
  }

  PVideoFrame OffsetPadFrame(PVideoFrame& frame, IScriptEnvironment* env)
  {
    int vpad = VPAD;
    int vpadUV = VPAD >> logUVy;

    return env->SubframePlanar(frame,
      frame->GetPitch(PLANAR_Y) * vpad, frame->GetPitch(PLANAR_Y), frame->GetRowSize(PLANAR_Y), frame->GetHeight(PLANAR_Y) - vpad * 2,
      frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U) * vpadUV, frame->GetPitch(PLANAR_U));
  }

public:
  KRemoveCombe(PClip clip, float thsmooth, float smooth, float thcombe, float ratio1, float ratio2, bool outcombe, bool show, IScriptEnvironment* env)
    : GenericVideoFilter(clip)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , padvi(vi)
    , thsmooth(thsmooth)
    , smooth(smooth)
    , thcombe(thcombe)
    , ratio1(ratio1)
    , ratio2(ratio2)
    , outcombe(outcombe)
    , show(show)
  {
    if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

    padvi.height += VPAD * 2;

    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);

    int flag_bytes = sizeof(uint8_t) * nBlkX * nBlkY;
    blockvi.pixel_type = VideoInfo::CS_BGR32;
    blockvi.width = 2048;
    blockvi.height = nblocks(flag_bytes, blockvi.width * 4);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame dst = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame flag = env->NewVideoFrame(vi);
    PVideoFrame flage = env->NewVideoFrame(vi);
    PVideoFrame blocks = env->NewVideoFrame(blockvi);

    CopyFrame(src, padded);
    PadFrame(padded);
    FindCombe3(padded, flag, (int)thsmooth);
    ExtendFlag(flag, flage);
    MergeUVFlags(flage);
    ApplyUVFlags(flage);
    RemoveCombe(padded, flage, dst, (int)smooth);

    if (outcombe) {
      PadFrame(dst);
      FindCombe3(dst, flag, (int)thcombe);
      MergeUVFlags(flag);
      CountFlags(blocks, flag, (int)ratio1, (int)ratio2);
      CleanBlocks(blocks);
      ExtendBlocks(blocks);
    }

    if (show) {
      VisualizeFlags(padded, flag);
      if (outcombe) {
        padded->SetProps(COMBE_FLAG_STR, blocks);
      }
      return padded;
    }

    if (outcombe) {
      dst->SetProps(COMBE_FLAG_STR, blocks);
    }
    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KRemoveCombe(
      args[0].AsClip(),       // source
      (float)args[1].AsFloat(30), // thsmooth
      (float)args[2].AsFloat(50), // smooth
      (float)args[3].AsFloat(150), // thcombe
      (float)args[4].AsFloat(0), // ratio1
      (float)args[5].AsFloat(5), // ratio2
      args[6].AsBool(true), // outcombe
      args[7].AsBool(false), // show
      env
    );
  }
};

class KShowCombe : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  int logUVx;
  int logUVy;
  int nBlkX, nBlkY;

  void ShowCombe(PVideoFrame& src, PVideoFrame& flag, PVideoFrame& dst)
  {
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));
    const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;
    int overlapUVx = OVERLAP >> logUVx;
    int overlapUVy = OVERLAP >> logUVy;

    int blue[] = { 73, 230, 111 };

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        int yStart = by * OVERLAP;
        int yEnd = yStart + OVERLAP;
        int xStart = bx * OVERLAP;
        int xEnd = xStart + OVERLAP;
        int yStartUV = by * overlapUVy;
        int yEndUV = yStartUV + overlapUVy;
        int xStartUV = bx * overlapUVx;
        int xEndUV = xStartUV + overlapUVx;

        bool isCombe = flagp[bx + by * nBlkX] != 0;

        for (int y = yStart; y < yEnd; ++y) {
          for (int x = xStart; x < xEnd; ++x) {
            dstY[x + y * pitchY] = isCombe ? blue[0] : srcY[x + y * pitchY];
          }
        }

        for (int y = yStartUV; y < yEndUV; ++y) {
          for (int x = xStartUV; x < xEndUV; ++x) {
            dstU[x + y * pitchUV] = isCombe ? blue[1] : srcU[x + y * pitchUV];
            dstV[x + y * pitchUV] = isCombe ? blue[2] : srcV[x + y * pitchUV];
          }
        }
      }
    }
  }
public:
  KShowCombe(PClip rc, IScriptEnvironment* env)
    : GenericVideoFilter(rc)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame flag = src->GetProps(COMBE_FLAG_STR)->GetFrame();
    PVideoFrame dst = env->NewVideoFrame(vi);

    ShowCombe(src, flag, dst);

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KShowCombe(
      args[0].AsClip(),       // source
      env
    );
  }
};

enum KFMSWTICH_FLAG {
  FRAME_60 = 1,
  FRAME_24,
};

class KFMSwitch : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  PClip clip24;
  PClip fmclip;
  PClip combeclip;
  float thresh;
  bool show;

  int logUVx;
  int logUVy;
  int nBlkX, nBlkY;

  PulldownPatterns patterns;

  bool ContainsDurtyBlock(PVideoFrame& flag)
  {
    const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        if (flagp[bx + by * nBlkX]) return true;
      }
    }

    return false;
  }

  void MergeBlock(PVideoFrame& src24, PVideoFrame& src60, PVideoFrame& flag, PVideoFrame& dst)
  {
    const pixel_t* src24Y = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_Y));
    const pixel_t* src24U = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_U));
    const pixel_t* src24V = reinterpret_cast<const pixel_t*>(src24->GetReadPtr(PLANAR_V));
    const pixel_t* src60Y = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_Y));
    const pixel_t* src60U = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_U));
    const pixel_t* src60V = reinterpret_cast<const pixel_t*>(src60->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));
    const uint8_t* flagp = reinterpret_cast<const uint8_t*>(flag->GetReadPtr());

    int pitchY = src24->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src24->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;
    int overlapUVx = OVERLAP >> logUVx;
    int overlapUVy = OVERLAP >> logUVy;

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        int yStart = by * OVERLAP;
        int yEnd = yStart + OVERLAP;
        int xStart = bx * OVERLAP;
        int xEnd = xStart + OVERLAP;
        int yStartUV = by * overlapUVy;
        int yEndUV = yStartUV + overlapUVy;
        int xStartUV = bx * overlapUVx;
        int xEndUV = xStartUV + overlapUVx;

        bool isCombe = flagp[bx + by * nBlkX] != 0;

        for (int y = yStart; y < yEnd; ++y) {
          for (int x = xStart; x < xEnd; ++x) {
            dstY[x + y * pitchY] = (isCombe ? src60Y : src24Y)[x + y * pitchY];
          }
        }

        for (int y = yStartUV; y < yEndUV; ++y) {
          for (int x = xStartUV; x < xEndUV; ++x) {
            dstU[x + y * pitchUV] = (isCombe ? src60U : src24U)[x + y * pitchUV];
            dstV[x + y * pitchUV] = (isCombe ? src60V : src24V)[x + y * pitchUV];
          }
        }
      }
    }
  }

  PVideoFrame InternalGetFrame(int n60, PVideoFrame& fmframe, int& type, IScriptEnvironment* env)
  {
    int cycleIndex = n60 / 10;
    const std::pair<int, float>* pfm = (std::pair<int, float>*)fmframe->GetReadPtr();

    if (pfm->second > thresh) {
      // コストが高いので60pと判断
      PVideoFrame frame60 = child->GetFrame(n60, env);
      type = FRAME_60;
      return frame60;
    }

    type = FRAME_24;

    // 24pフレーム番号を取得
    Frame24InfoV2 frameInfo = patterns.GetFrame60(pfm->first, n60);
    int n24 = frameInfo.cycleIndex * 4 + frameInfo.frameIndex;

    if (frameInfo.frameIndex < 0) {
      // 前に空きがあるので前のサイクル
      n24 = frameInfo.cycleIndex * 4 - 1;
    }
    else if (frameInfo.frameIndex >= 4) {
      // 後ろのサイクルのパターンを取得
      PVideoFrame nextfmframe = fmclip->GetFrame(cycleIndex + 1, env);
      const std::pair<int, float>* pnextfm = (std::pair<int, float>*)nextfmframe->GetReadPtr();
      int fstart = patterns.GetFrame24(pnextfm->first, 0).fieldStartIndex;
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
    PVideoFrame flag = combeclip->GetFrame(n24, env)->GetProps(COMBE_FLAG_STR)->GetFrame();

    if (ContainsDurtyBlock(flag) == false) {
      // ダメなブロックはないのでそのまま返す
      return frame24;
    }

    // ダメなブロックは60pフレームからコピー
    PVideoFrame frame60 = child->GetFrame(n60, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    MergeBlock(frame24, frame60, flag, dst);

    return dst;
  }

  void DrawInfo(PVideoFrame& dst, const char* fps, int pattern, float score, IScriptEnvironment* env) {
    env->MakeWritable(&dst);

    char buf[100]; sprintf(buf, "KFMSwitch: %s pattern:%2d cost:%.1f", fps, pattern, score);
    DrawText(dst, true, 0, 0, buf);
  }

public:
  KFMSwitch(PClip clip60, PClip clip24, PClip fmclip, PClip combeclip, float thresh, bool show, IScriptEnvironment* env)
    : GenericVideoFilter(clip60)
    , clip24(clip24)
    , fmclip(fmclip)
    , combeclip(combeclip)
    , thresh(thresh)
    , show(show)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    if (vi.width & 7) env->ThrowError("[KFMSwitch]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KFMSwitch]: height must be multiple of 8");

    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);
  }

  PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env)
  {
    int cycleIndex = n60 / 10;
    PVideoFrame fmframe = fmclip->GetFrame(cycleIndex, env);
    int frameType;

    PVideoFrame dst = InternalGetFrame(n60, fmframe, frameType, env);

    if (show) {
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
      (float)args[4].AsFloat(),   // thresh
      args[5].AsBool(),           // show
      env
    );
  }
};

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KShowStatic", "cc", KShowStatic::Create, 0);
  env->AddFunction("KMergeStatic", "ccc", KMergeStatic::Create, 0);

  env->AddFunction("KFMFrameDev", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameDev::Create, 0);
  env->AddFunction("KFMFrameAnalyze", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyze::Create, 0);
  env->AddFunction("KFMCycleAnalyze", "cc", KFMCycleAnalyze::Create, 0);
  env->AddFunction("KTelecine", "cc[show]b", KTelecine::Create, 0);
  env->AddFunction("KRemoveCombe", "c[thsmooth]f[smooth]f[thcombe]f[ratio1]f[ratio2]f[outcombe]b[show]b", KRemoveCombe::Create, 0);
  env->AddFunction("KShowCombe", "c", KShowCombe::Create, 0);
  env->AddFunction("KFMSwitch", "cccc[thresh]f[show]b", KFMSwitch::Create, 0);
}

#include <Windows.h>

void AddFuncFMKernel(IScriptEnvironment* env);

static void init_console()
{
  AllocConsole();
  freopen("CONOUT$", "w", stdout);
  freopen("CONIN$", "r", stdin);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;
  //init_console();

  AddFuncFM(env);
  AddFuncFMKernel(env);

  return "K Field Matching Plugin";
}
