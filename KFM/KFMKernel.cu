
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

class KPatchCombe : public KFMFilterBase
{
  PClip clip60;
  PClip combemaskclip;
  PClip containscombeclip;
  PClip fmclip;

  PulldownPatterns patterns;

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    PDevice cpuDevice = env->GetDevice(DEV_TYPE_CPU, 0);

    {
      Frame containsframe = env->GetFrame(containscombeclip, n, cpuDevice);
      if (*containsframe.GetReadPtr<int>() == 0) {
        // ダメなブロックはないのでそのまま返す
        return child->GetFrame(n, env);
      }
    }

    int cycleIndex = n / 4;
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpuDevice)).GetReadPtr<KFMResult>());
    Frame24Info frameInfo = patterns.GetFrame24(fm.pattern, n);

    int fieldIndex[] = { 1, 3, 6, 8 };
    // 標準位置
    int n60 = fieldIndex[n % 4];
    // フィールド対象範囲に補正
    n60 = clamp(n60, frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1);
    n60 += cycleIndex * 10;

    Frame baseFrame = child->GetFrame(n, env);
    Frame frame60 = clip60->GetFrame(n60, env);
    Frame mflag = combemaskclip->GetFrame(n, env);

    // ダメなブロックはbobフレームからコピー
    Frame dst = env->NewVideoFrame(vi);
    MergeBlock<pixel_t>(baseFrame, frame60, mflag, dst, env);

    return dst.frame;
  }

public:
  KPatchCombe(PClip clip24, PClip clip60, PClip fmclip, PClip combemaskclip, PClip containscombeclip, IScriptEnvironment* env)
    : KFMFilterBase(clip24)
    , clip60(clip60)
    , combemaskclip(combemaskclip)
    , containscombeclip(containscombeclip)
    , fmclip(fmclip)
  {
    // チェック
    CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);
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
      env->ThrowError("[KPatchCombe] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KPatchCombe(
      args[0].AsClip(),       // clip24
      args[1].AsClip(),       // clip60
      args[2].AsClip(),       // fmclip
      args[3].AsClip(),       // combemaskclip
      args[4].AsClip(),       // containscombeclip
      env
    );
  }
};

enum KFMSWTICH_FLAG {
  FRAME_60 = 1,
  FRAME_30,
	FRAME_24,
  FRAME_UCF,
};

class KFMSwitch : public KFMFilterBase
{
	typedef uint8_t pixel_t;

  enum Mode {
    NORMAL = 0,
    WITH_FRAME_DURATION = 1,
    ONLY_FRAME_DURATION = 2,
  };

  VideoInfo srcvi;

  PClip clip24;
  PClip mask24;
  PClip cc24;
  
  PClip clip30;
  PClip mask30;
  PClip cc30;

	PClip fmclip;
  PClip combemaskclip;
  PClip containscombeclip;
  PClip ucfclip;
	float thswitch;
  int mode; // 0:通常 1:通常(FrameDurationあり) 2:FrameDurationのみ
	bool show;
	bool showflag;

  int analyzeMode;

	int logUVx;
	int logUVy;
	int nBlkX, nBlkY;

	bool is30_60;
	bool is24_60;

	PulldownPatterns patterns;

	template <typename pixel_t>
	void VisualizeFlag(Frame& dst, Frame& flag, PNeoEnv env)
	{
		// 判定結果を表示
		int blue[] = { 73, 230, 111 };

		pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
		pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
		pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);
    const uint8_t* flagY = flag.GetReadPtr<uint8_t>(PLANAR_Y);
    const uint8_t* flagC = flag.GetReadPtr<uint8_t>(PLANAR_U);

		int dstPitchY = dst.GetPitch<pixel_t>(PLANAR_Y);
		int dstPitchUV = dst.GetPitch<pixel_t>(PLANAR_U);
    int fpitchY = flag.GetPitch<uint8_t>(PLANAR_Y);
    int fpitchUV = flag.GetPitch<uint8_t>(PLANAR_U);

		// 色を付ける
		for (int y = 0; y < srcvi.height; ++y) {
			for (int x = 0; x < srcvi.width; ++x) {
        int coefY = flagY[x + y * fpitchY];
				int offY = x + y * dstPitchY;
        dstY[offY] = (blue[0] * coefY + dstY[offY] * (128 - coefY)) >> 7;
        
        int coefC = flagC[(x >> logUVx) + (y >> logUVy) * fpitchUV];
				int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
				dstU[offUV] = (blue[1] * coefC + dstU[offUV] * (128 - coefC)) >> 7;
				dstV[offUV] = (blue[2] * coefC + dstV[offUV] * (128 - coefC)) >> 7;
			}
		}
	}

  struct FrameInfo {
    int baseType;
    int maskType;
    int n24;
  };

  Frame GetBaseFrame(int n60, FrameInfo& info, PNeoEnv env)
  {
    switch (info.baseType) {
    case FRAME_60:
      return child->GetFrame(n60, env);
    case FRAME_UCF:
      return ucfclip->GetFrame(n60, env);
    case FRAME_30:
			return clip30->GetFrame(is30_60 ? n60 : (n60 >> 1), env);
    case FRAME_24:
      return clip24->GetFrame(is24_60 ? n60 : info.n24, env);
    }
    return Frame();
  }

  Frame GetMaskFrame(int n60, FrameInfo& info, PNeoEnv env)
  {
    switch (info.maskType) {
    case FRAME_30:
      return mask30->GetFrame(n60 >> 1, env);
    case FRAME_24:
      return mask24->GetFrame(info.n24, env);
    }
    return Frame();
  }
  
	template <typename pixel_t>
	Frame InternalGetFrame(int n60, FrameInfo& info, PNeoEnv env)
	{
    Frame baseFrame = GetBaseFrame(n60, info, env);
    if (info.maskType == 0) {
      return baseFrame;
    }

    Frame mflag = GetMaskFrame(n60, info, env);
    Frame frame60 = child->GetFrame(n60, env);

		if (!IS_CUDA && srcvi.ComponentSize() == 1 && showflag) {
			env->MakeWritable(&baseFrame.frame);
			VisualizeFlag<pixel_t>(baseFrame, mflag, env);
			return baseFrame;
		}

		// ダメなブロックはbobフレームからコピー
		Frame dst = env->NewVideoFrame(srcvi);
		MergeBlock<pixel_t>(baseFrame, frame60, mflag, dst, env);

		return dst;
  }

	FrameInfo GetFrameInfo(int n60, KFMResult fm, PNeoEnv env)
	{
		int cycleIndex = n60 / 10;
    Frame baseFrame;
    FrameInfo info = { 0 };

    // 60p判定は 1パスの場合はコスト 2パスの場合はKFMCycleAnalyzeの結果 を使う
		if ((analyzeMode == 0 && fm.cost > thswitch) || (analyzeMode != 0 && fm.is60p)) {
			// コストが高いので60pと判断
      info.baseType = ucfclip ? FRAME_UCF : FRAME_60;

      if (mode == ONLY_FRAME_DURATION) {
        // FrameDurationのみならUCFと60を区別する必要はないので
        // フレームの生成を避けるためここで帰る
        return info;
      }

      if (ucfclip) {
        // フレームにアクセスが発生するので注意
        // ここでは60fpsに決定してるので、
        // 次のGetFrameでこのフレームが必要なことは決定している
        baseFrame = ucfclip->GetFrame(n60, env);
        auto prop = baseFrame.GetProperty(DECOMB_UCF_FLAG_STR);
        if (prop == nullptr) {
          env->ThrowError("Invalid UCF clip");
        }
        auto flag = (DECOMB_UCF_FLAG)prop->GetInt();
        // フレーム置換がされた場合は、60p部分マージ処理を実行する
        if (flag != DECOMB_UCF_NEXT && flag != DECOMB_UCF_PREV) {
          return info;
        }
      }
      else {
        return info;
      }
		}

    // ここでのtypeは 24 or 30 or UCF

    if (PulldownPatterns::Is30p(fm.pattern)) {
      // 30p
      int n30 = n60 >> 1;

      if (!baseFrame) {
        info.baseType = FRAME_30;
      }

      Frame containsframe = env->GetFrame(cc30, n30, env->GetDevice(DEV_TYPE_CPU, 0));
      info.maskType = *containsframe.GetReadPtr<int>() ? FRAME_30 : 0;
    }
    else {
      // 24pフレーム番号を取得
      Frame24Info frameInfo = patterns.GetFrame60(fm.pattern, n60);
      // fieldShiftでサイクルをまたぐこともあるので、frameIndexはfieldShift込で計算
      int frameIndex = frameInfo.frameIndex + frameInfo.fieldShift;
      int n24 = frameInfo.cycleIndex * 4 + frameIndex;

      if (frameIndex < 0) {
        // 前に空きがあるので前のサイクル
        n24 = frameInfo.cycleIndex * 4 - 1;
      }
      else if (frameIndex >= 4) {
        // 後ろのサイクルのパターンを取得
        PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
        auto nextfm = *(Frame(env->GetFrame(fmclip, cycleIndex + 1, cpudev)).GetReadPtr<KFMResult>());
        int fstart = patterns.GetFrame24(nextfm.pattern, 0).fieldStartIndex;
        if (fstart > 0) {
          // 前に空きがあるので前のサイクル
          n24 = frameInfo.cycleIndex * 4 + 3;
        }
        else {
          // 前に空きがないので後ろのサイクル
          n24 = frameInfo.cycleIndex * 4 + 4;
        }
      }

      if (!baseFrame) {
        info.baseType = FRAME_24;
      }

      Frame containsframe = env->GetFrame(cc24, n24, env->GetDevice(DEV_TYPE_CPU, 0));
      info.maskType = *containsframe.GetReadPtr<int>() ? FRAME_24 : 0;
      info.n24 = n24;
    }

		return info;
	}

  static const char* FrameTypeStr(int frameType)
  {
    switch (frameType) {
    case FRAME_60: return "60p";
    case FRAME_30: return "30p";
    case FRAME_24: return "24p";
    case FRAME_UCF: return "UCF";
    }
    return "???";
  }

  int GetFrameDuration(int n60, FrameInfo& info, PNeoEnv env)
  {
    int duration = 1;
    // 60fpsマージ部分がある場合は60fps
    if (info.maskType == 0) {
      int source;
      // 最大durationを設定
      switch (info.baseType) {
      case FRAME_60:
      case FRAME_UCF:
        duration = 1;
        source = n60;
        break;
      case FRAME_30:
        duration = 2;
        source = n60 >> 1;
        break;
      case FRAME_24:
        duration = 4;
        source = info.n24;
        break;
      }
      PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
      for (int i = 1; i < duration; ++i) {
        // ここでは FRAME_30 or FRAME_24
        if (n60 + i >= vi.num_frames) {
          // フレーム数を超えてる
          duration = i;
          break;
        }
        int cycleIndex = (n60 + i) / 10;
        KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpudev)).GetReadPtr<KFMResult>());
        FrameInfo next = GetFrameInfo(n60 + i, fm, env);
        if (next.baseType != info.baseType) {
          // ベースタイプが違ったら同じフレームでない
          duration = i;
          break;
        }
        else {
          int nextsource = -1;
          switch (next.baseType) {
          case FRAME_30:
            nextsource = (n60 + i) >> 1;
            break;
          case FRAME_24:
            nextsource = next.n24;
            break;
          }
          if (nextsource != source) {
            // ソースフレームが違ったら同じフレームでない
            duration = i;
            break;
          }
        }
      }
    }
    return duration;
  }

  template <typename pixel_t>
  PVideoFrame GetFrameTop(int n60, PNeoEnv env)
  {
    PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
    int cycleIndex = n60 / 10;
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpudev)).GetReadPtr<KFMResult>());
    FrameInfo info = GetFrameInfo(n60, fm, env);
    
    Frame dst;
    if (mode != ONLY_FRAME_DURATION) {
      dst = InternalGetFrame<pixel_t>(n60, info, env);
    }
    else {
      dst = env->NewVideoFrame(srcvi);
    }

    int duration = 0;
    if (mode != NORMAL) {
      duration = GetFrameDuration(n60, info, env);
      dst.SetProperty("FrameDuration", duration);
    }

    if (show) {
      const char* fps = FrameTypeStr(info.baseType);
      char buf[100]; sprintf(buf, "KFMSwitch: %s dur: %d pattern:%2d cost:%.3f", fps, duration, fm.pattern, fm.cost);
      DrawText<pixel_t>(dst.frame, srcvi.BitsPerComponent(), 0, 0, buf, env);
      return dst.frame;
    }

    return dst.frame;
  }

public:
	KFMSwitch(PClip clip60, PClip fmclip,
    PClip clip24, PClip mask24, PClip cc24,
    PClip clip30, PClip mask30, PClip cc30,
    PClip ucfclip,
		float thswitch, int mode, bool show, bool showflag, IScriptEnvironment* env)
		: KFMFilterBase(clip60)
    , srcvi(vi)
    , fmclip(fmclip)
    , clip24(clip24)
    , mask24(mask24)
    , cc24(cc24)
    , clip30(clip30)
    , mask30(mask30)
    , cc30(cc30)
    , ucfclip(ucfclip)
		, thswitch(thswitch)
    , mode(mode)
		, show(show)
		, showflag(showflag)
		, logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
		, logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
	{
		if (vi.width & 7) env->ThrowError("[KFMSwitch]: width must be multiple of 8");
		if (vi.height & 7) env->ThrowError("[KFMSwitch]: height must be multiple of 8");

		nBlkX = nblocks(vi.width, OVERLAP);
		nBlkY = nblocks(vi.height, OVERLAP);

    if (mode < 0 || mode > 2) {
      env->ThrowError("[KFMSwitch] mode(%d) must be in range 0-2", mode);
    }

    auto info = CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);
    analyzeMode = info->mode;

    // check clip device
    if (!(GetDeviceTypes(fmclip) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: fmclip must be CPU device");
    }
    if (!(GetDeviceTypes(cc24) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: cc24 must be CPU device");
    }
    if (!(GetDeviceTypes(cc30) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: cc30 must be CPU device");
    }

    auto devs = GetDeviceTypes(clip60);
    if (!(GetDeviceTypes(clip24) & devs)) {
      env->ThrowError("[KFMSwitch]: clip24 device unmatch");
    }
    if (!(GetDeviceTypes(clip30) & devs)) {
      env->ThrowError("[KFMSwitch]: clip30 device unmatch");
    }
    if (!(GetDeviceTypes(mask24) & devs)) {
      env->ThrowError("[KFMSwitch]: mask24 device unmatch");
    }
    if (!(GetDeviceTypes(mask30) & devs)) {
      env->ThrowError("[KFMSwitch]: mask30 device unmatch");
    }
    if (ucfclip && !(GetDeviceTypes(ucfclip) & devs)) {
      env->ThrowError("[KFMSwitch]: ucfclip device unmatch");
    }

    // VideoInfoチェック
    VideoInfo vi60 = clip60->GetVideoInfo();
    VideoInfo vifm = fmclip->GetVideoInfo();
    VideoInfo vi24 = clip24->GetVideoInfo();
    VideoInfo vimask24 = mask24->GetVideoInfo();
    VideoInfo vicc24 = cc24->GetVideoInfo();
    VideoInfo vi30 = clip30->GetVideoInfo();
    VideoInfo vimask30 = mask30->GetVideoInfo();
    VideoInfo vicc30 = cc30->GetVideoInfo();
    VideoInfo viucf = ucfclip ? ucfclip->GetVideoInfo() : VideoInfo();

		// 24/30クリップは補間された60fpsか見る
		is24_60 = (vi24.fps_numerator == vi60.fps_numerator) && (vi24.fps_denominator == vi60.fps_denominator);
		is30_60 = (vi30.fps_numerator == vi60.fps_numerator) && (vi30.fps_denominator == vi60.fps_denominator);

    // fpsチェック
		if (is24_60 == false) {
			if (vi24.fps_denominator != vimask24.fps_denominator)
				env->ThrowError("[KFMSwitch]: vi24.fps_denominator != vimask24.fps_denominator");
			if (vi24.fps_numerator != vimask24.fps_numerator)
				env->ThrowError("[KFMSwitch]: vi24.fps_numerator != vimask24.fps_numerator");
		}
		if (vicc24.fps_denominator != vimask24.fps_denominator)
			env->ThrowError("[KFMSwitch]: vicc24.fps_denominator != vimask24.fps_denominator");
		if (vicc24.fps_numerator != vimask24.fps_numerator)
			env->ThrowError("[KFMSwitch]: vicc24.fps_numerator != vimask24.fps_numerator");
		if (is30_60 == false) {
			if (vi30.fps_denominator != vimask30.fps_denominator)
				env->ThrowError("[KFMSwitch]: vi30.fps_denominator != vimask30.fps_denominator");
			if (vi30.fps_numerator != vimask30.fps_numerator)
				env->ThrowError("[KFMSwitch]: vi30.fps_numerator != vimask30.fps_numerator");
		}
		if (vicc30.fps_denominator != vimask30.fps_denominator)
			env->ThrowError("[KFMSwitch]: vicc30.fps_denominator != vimask30.fps_denominator");
		if (vicc30.fps_numerator != vimask30.fps_numerator)
			env->ThrowError("[KFMSwitch]: vicc30.fps_numerator != vimask30.fps_numerator");
    if (ucfclip) {
      if (vi60.fps_denominator != viucf.fps_denominator)
        env->ThrowError("[KFMSwitch]: vi60.fps_denominator != viucf.fps_denominator");
      if (vi60.fps_numerator != viucf.fps_numerator)
        env->ThrowError("[KFMSwitch]: vi60.fps_numerator != viucf.fps_numerator");
    }

    // サイズチェック
    if (vi60.width != vi24.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vi24.width");
    if (vi60.height != vi24.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vi24.height");
    if (vi60.width != vimask24.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vimask24.width");
    if (vi60.height != vimask24.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vimask24.height");
    if (vi60.width != vi30.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vi30.width");
    if (vi60.height != vi30.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vi30.height");
    if (vi60.width != vimask30.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vimask30.width");
    if (vi60.height != vimask30.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vimask30.height");
    if (ucfclip) {
      if (vi60.width != viucf.width)
        env->ThrowError("[KFMSwitch]: vi60.width != viucf.width");
      if (vi60.height != viucf.height)
        env->ThrowError("[KFMSwitch]: vi60.height != viucf.height");
    }

    // UCFクリップチェック
    if (ucfclip) {
      if(DecombUCFInfo::GetParam(viucf, env)->fpsType != 60)
        env->ThrowError("[KFMSwitch]: Invalid UCF clip (KDecombUCF60 clip is required)");
    }
	}

	PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
	{
		PNeoEnv env = env_;

		int pixelSize = srcvi.ComponentSize();
		switch (pixelSize) {
		case 1:
			return GetFrameTop<uint8_t>(n60, env);
		case 2:
      return GetFrameTop<uint16_t>(n60, env);
		default:
			env->ThrowError("[KFMSwitch] Unsupported pixel format");
			break;
		}

		return PVideoFrame();
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KFMSwitch(
			args[0].AsClip(),           // clip60
      args[1].AsClip(),           // fmclip
      args[2].AsClip(),           // clip24
      args[3].AsClip(),           // mask24
      args[4].AsClip(),           // cc24
      args[5].AsClip(),           // clip30
			args[6].AsClip(),           // mask30
      args[7].AsClip(),           // cc30
      args[8].Defined() ? args[8].AsClip() : nullptr,           // ucfclip
      (float)args[9].AsFloat(3.0f),// thswitch
      args[10].AsInt(0),           // mode
      args[11].AsBool(false),      // show
			args[12].AsBool(false),      // showflag
			env
			);
	}
};

class KFMPad : public KFMFilterBase
{
  VideoInfo srcvi;

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame src = child->GetFrame(n, env);
    Frame dst = Frame(env->NewVideoFrame(vi), VPAD);

    CopyFrame<pixel_t>(src, dst, env);
    PadFrame<pixel_t>(dst, env);

    return dst.frame;
  }
public:
  KFMPad(PClip src, IScriptEnvironment* env)
    : KFMFilterBase(src)
    , srcvi(vi)
  {
    if (srcvi.width & 3) env->ThrowError("[KFMPad]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KFMPad]: height must be multiple of 4");

    vi.height += VPAD * 2;
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
      env->ThrowError("[KFMPad] Unsupported pixel format");
      break;
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMPad(
      args[0].AsClip(),       // src
      env
    );
  }
};


class AssumeDevice : public GenericVideoFilter
{
  int devices;
public:
  AssumeDevice(PClip clip, int devices)
    : GenericVideoFilter(clip)
    , devices(devices)
  { }

	int __stdcall SetCacheHints(int cachehints, int frame_range) {
		if (cachehints == CACHE_GET_DEV_TYPE) {
			return devices;
		}
		return 0;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new AssumeDevice(args[0].AsClip(), args[1].AsInt());
	}
};

void AddFuncFMKernel(IScriptEnvironment* env)
{
  env->AddFunction("KPatchCombe", "ccccc", KPatchCombe::Create, 0);
  env->AddFunction("KFMSwitch", "cccccccc[ucfclip]c[thswitch]f[mode]i[show]b[showflag]b", KFMSwitch::Create, 0);
  env->AddFunction("KFMPad", "c", KFMPad::Create, 0);
	env->AddFunction("AssumeDevice", "ci", AssumeDevice::Create, 0);
}
