
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>

#include "CommonFunctions.h"
#include "TextOut.h"
#include "Frame.h"
#include "KMV.h"
#include "KFM.h"
#include "Copy.h"

void OnCudaError(cudaError_t err) {
#if 1 // デバッグ用（本番は取り除く）
  printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
#endif
}

int GetDeviceTypes(const PClip& clip)
{
  int devtypes = (clip->GetVersion() >= 5) ? clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  if (devtypes == 0) {
    return DEV_TYPE_CPU;
  }
  return devtypes;
}

int GetCommonDevices(const std::vector<PClip>& clips)
{
  int devs = DEV_TYPE_ANY;
  for (const PClip& c : clips) {
    if (c) {
      devs &= GetDeviceTypes(c);
    }
  }
  return devs;
}

class KShowStatic : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  PClip sttclip;

  int logUVx;
  int logUVy;

  void CopyFrame(Frame& src, Frame& dst, PNeoEnv env)
  {
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

    Copy(dstY, pitchY, srcY, pitchY, vi.width, vi.height, env);
    Copy(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV, env);
    Copy(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV, env);
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

  void VisualizeBlock(Frame& flag, Frame& dst)
  {
    const pixel_t* flagY = flag.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* flagU = flag.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* flagV = flag.GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

    int flagPitchY = flag.GetPitch<uint8_t>(PLANAR_Y);
    int flagPitchUV = flag.GetPitch<uint8_t>(PLANAR_U);
    int dstPitchY = dst.GetPitch<pixel_t>(PLANAR_Y);
    int dstPitchUV = dst.GetPitch<pixel_t>(PLANAR_U);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    int blue[] = { 73, 230, 111 };

    MaskFill(dstY, dstPitchY, flagY, flagPitchY, vi.width, vi.height, blue[0]);
    MaskFill(dstU, dstPitchUV, flagU, flagPitchUV, widthUV, heightUV, blue[1]);
    MaskFill(dstV, dstPitchUV, flagV, flagPitchUV, widthUV, heightUV, blue[2]);
  }

public:
  KShowStatic(PClip sttclip, PClip clip30, PNeoEnv env)
    : GenericVideoFilter(clip30)
    , sttclip(sttclip)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame flag = sttclip->GetFrame(n, env);
    Frame frame30 = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

    CopyFrame(frame30, dst, env);
    VisualizeBlock(flag, dst);

    return dst.frame;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KShowStatic(
      args[0].AsClip(),       // sttclip
      args[1].AsClip(),       // clip30
      env);
  }
};

struct FMData {
	// 縞と動きの和
	float mft[14];
	float mftr[14];
	float mftcost[14];
};

float RSplitScore(const PulldownPatternField* pattern, const float* fv) {
	float sumsplit = 0, sumnsplit = 0;

	for (int i = 0; i < 14; ++i) {
		if (pattern[i].split) {
			sumsplit += fv[i];
		}
		else {
			sumnsplit += fv[i];
		}
	}

	return sumsplit - sumnsplit;
}

float RSplitCost(const PulldownPatternField* pattern, const float* fv, const float* fvcost, float costth) {
	int nsplit = 0;
	float sumcost = 0;

	for (int i = 0; i < 14; ++i) {
		if (pattern[i].split) {
			nsplit++;
			if (fv[i] < costth) {
				sumcost += (costth - fv[i]) * fvcost[i];
			}
		}
	}

	return sumcost / nsplit;
}

#pragma region PulldownPatterns

PulldownPattern::PulldownPattern(int nf0, int nf1, int nf2, int nf3)
  : fields()
  , cycle(10)
{
  // 24p
  if (nf0 + nf1 + nf2 + nf3 != 10) {
    printf("Error: sum of nfields must be 10.\n");
  }
  if (nf0 == nf2 && nf1 == nf3) {
    cycle = 5;
  }
  int nfields[] = { nf0, nf1, nf2, nf3 };
  for (int c = 0, fstart = 0; c < 4; ++c) {
    for (int i = 0; i < 4; ++i) {
      int nf = nfields[i];
      for (int f = 0; f < nf - 2; ++f) {
        fields[fstart + f].merge = true;
      }
      fields[fstart + nf - 1].split = true;

      // 縞なし24fps対応
      if (nf0 == 4 && nf1 == 2 && nf2 == 2 && nf3 == 2) {
        if (i < 2) {
          fields[fstart + nf - 1].shift = true;
        }
      }

      fstart += nf;
    }
  }
}

PulldownPattern::PulldownPattern()
	: fields()
  , cycle(2)
{
  // 30p
	for (int c = 0, fstart = 0; c < 4; ++c) {
		for (int i = 0; i < 4; ++i) {
			int nf = 2;
			fields[fstart + nf - 1].split = true;
			fstart += nf;
		}
	}
}

PulldownPatterns::PulldownPatterns()
  : p2323(2, 3, 2, 3)
  , p2233(2, 2, 3, 3)
  , p2224(4, 2, 2, 2)
  , p30()
{
  const PulldownPattern* patterns[] = { &p2323, &p2233, &p2224, &p30 };
  int steps[] = { 1, 1, 2, 2 };

  int pi = 0;
  for (int p = 0; p < 4; ++p) {
    patternOffsets[p] = pi;
    for (int i = 0; i < patterns[p]->GetCycleLength(); i += steps[p]) {
      allpatterns[pi++] = patterns[p]->GetPattern(i);
    }
  }
  patternOffsets[4] = pi;

  if (pi != NUM_PATTERNS) {
    throw "Error !!!";
  }
}

const char* PulldownPatterns::PatternToString(int patternIndex, int& index) const
{
  const char* names[] = { "2323", "2233", "2224", "30p" };
  auto pattern = std::upper_bound(patternOffsets, patternOffsets + 4, patternIndex) - patternOffsets - 1;
  index = patternIndex - patternOffsets[pattern];
  return names[pattern];
}

Frame24Info PulldownPatterns::GetFrame24(int patternIndex, int n24) const {
  Frame24Info info = { 0 };
  info.cycleIndex = n24 / 4;
  info.frameIndex = n24 % 4;

	int searchFrame = info.frameIndex;

	// パターンが30pの場合は、5枚中真ん中の1枚を消す
	// 30pの場合は、24pにした時点で5枚中1枚が失われてしまうので、正しく60pに復元することはできない
	// 30p部分は60pクリップから取得されるので基本的には問題ないが、
	// 前後のサイクルが24pで、サイクル境界の空きフレームとして30p部分も取得されることがある
	// なので、5枚中、最初と最後のフレームだけは正しく60pに復元する必要がある
	// 以下の処理がないと最後のフレーム(4枚目)がズレてしまう
	if (patternIndex >= patternOffsets[3]) {
		if (searchFrame >= 2) ++searchFrame;
	}

  const PulldownPatternField* ptn = allpatterns[patternIndex];
  int fldstart = 0;
  int nframes = 0;
  for (int i = 0; i < 16; ++i) {
    if (ptn[i].split) {
      if (fldstart >= 1) {
        if (nframes++ == searchFrame) {
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

Frame24Info PulldownPatterns::GetFrame60(int patternIndex, int n60) const {
  Frame24Info info = { 0 };
  info.cycleIndex = n60 / 10;

  // splitでフレームの切り替わりを見る
  // splitは前のフレームの最後のフィールドとなるので
  // -2フィールド目から見ると検出できる最初のフレームは
  // 最も早くて-1フィールドスタート
  // これを前提にアルゴリズムが組まれていることに注意

  const PulldownPatternField* ptn = allpatterns[patternIndex];
  int fldstart = 0;
  int nframes = -1;
  int findex = n60 % 10;

  // 最大4フィールドがあるので、2+10+4だけ回してる
  for (int i = 0; i < 16; ++i) {
    if (ptn[i].split) {
      if (fldstart >= 1) {
        ++nframes;
      }
      int nextfldstart = i + 1;
      if (findex < nextfldstart - 2) {
        info.frameIndex = nframes;
        info.fieldStartIndex = fldstart - 2;
        info.numFields = nextfldstart - fldstart;
        info.fieldShift = ptn[findex + 2].shift ? 1 : 0;
        return info;
      }
      fldstart = i + 1;
    }
  }

  throw "Error !!!";
}

std::pair<int, float> PulldownPatterns::Matching(const FMData* data, int width, int height, float costth, float adj2224, float adj30) const
{
	std::vector<float> mtshima(NUM_PATTERNS);
	std::vector<float> mtshimacost(NUM_PATTERNS);

  // 各スコアを計算
  for (int i = 0; i < NUM_PATTERNS; ++i) {
    mtshima[i] = RSplitScore(allpatterns[i], data->mftr);
    mtshimacost[i] = RSplitCost(allpatterns[i], data->mftr, data->mftcost, costth);
  }

  // 調整
  for (int i = patternOffsets[2]; i < patternOffsets[3]; ++i) {
    mtshima[i] -= adj2224;
  }
  for (int i = patternOffsets[3]; i < patternOffsets[4]; ++i) {
    mtshima[i] -= adj30;
  }

	auto makeRet = [&](int n) {
		float cost = mtshimacost[n];
		return std::pair<int, float>(n, cost);
	};

	auto it = std::max_element(mtshima.begin(), mtshima.end());
	return makeRet((int)(it - mtshima.begin()));
}

#pragma endregion

class KFMCycleAnalyze : public GenericVideoFilter
{
	PClip source;
  VideoInfo srcvi;
	PulldownPatterns patterns;
	float lscale;
	float costth;
  float adj2224;
  float adj30;
public:
  KFMCycleAnalyze(PClip fmframe, PClip source, float lscale, float costth, float adj2224, float adj30, IScriptEnvironment* env)
		: GenericVideoFilter(fmframe)
		, source(source)
    , srcvi(source->GetVideoInfo())
		, lscale(lscale)
		, costth(costth)
    , adj2224(adj2224)
    , adj30(adj30)
	{
    int out_bytes = sizeof(std::pair<int, float>);
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 4;
    vi.height = nblocks(out_bytes, vi.width * 4);
  }

  PVideoFrame __stdcall GetFrame(int cycle, IScriptEnvironment* env)
	{
		FMCount fmcnt[18];
		for (int i = -2; i <= 6; ++i) {
			Frame frame = child->GetFrame(cycle * 5 + i, env);
			memcpy(fmcnt + (i + 2) * 2, frame.GetReadPtr<uint8_t>(), sizeof(fmcnt[0]) * 2);
		}

		// shima, lshima, moveの画素数がマチマチなので大きさの違いによる重みの違いが出る
		// shima, lshimaをmoveに合わせる（平均が同じになるようにする）

		int mft[18] = { 0 };
		for (int i = 1; i < 17; ++i) {
			int split = std::min(fmcnt[i - 1].move, fmcnt[i].move);
			mft[i] = split + fmcnt[i].shima + (int)(fmcnt[i].lshima * lscale);
		}

		FMData data = { 0 };
		int vbase = (int)(srcvi.width * srcvi.height * 0.001f);
    for (int i = 0; i < 14; ++i) {
			data.mft[i] = (float)mft[i + 2];
			data.mftr[i] = (mft[i + 2] + vbase) * 2.0f / (mft[i + 1] + mft[i + 3] + vbase * 2.0f) - 1.0f;
			data.mftcost[i] = (float)(mft[i + 1] + mft[i + 3]) / vbase;
		}

		auto result = patterns.Matching(&data, srcvi.width, srcvi.height, costth, adj2224, adj30);

    Frame dst = env->NewVideoFrame(vi);
    uint8_t* dstp = dst.GetWritePtr<uint8_t>();
    memcpy(dstp, &result, sizeof(result));

		// フレームをCUDAに持っていった後、
    // CPUからも取得できるようにプロパティにも入れておく
    dst.SetProperty("KFM_Pattern", result.first);
    dst.SetProperty("KFM_Cost", result.second);

    return dst.frame;
	}

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMCycleAnalyze(
      args[0].AsClip(),       // fmframe
      args[1].AsClip(),       // source
			(float)args[2].AsFloat(5.0f), // lscale
      (float)args[3].AsFloat(1.0f), // costth
      (float)args[4].AsFloat(0.5f), // adj2224
      (float)args[5].AsFloat(0.5f), // adj30
      env
    );
  }
};

class Print : public GenericVideoFilter
{
  std::string str;
  int x, y;
public:
  Print(PClip clip, const char* str, int x, int y, IScriptEnvironment* env)
    : GenericVideoFilter(clip)
    , str(str)
    , x(x)
    , y(y)
  {
    int cs = vi.ComponentSize();
    if (cs != 1 && cs != 2)
      env->ThrowError("[Print] Unsupported pixel format");
    if(vi.IsRGB() || !vi.IsPlanar())
      env->ThrowError("[Print] Unsupported pixel format");
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PVideoFrame src = child->GetFrame(n, env);

    switch (vi.ComponentSize()) {
    case 1:
      DrawText<uint8_t>(src, vi.BitsPerComponent(), x, y, str, env);
      break;
    case 2:
      DrawText<uint16_t>(src, vi.BitsPerComponent(), x, y, str, env);
      break;
    }

    return src;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new Print(
      args[0].AsClip(),      // clip
      args[1].AsString(),    // str
      args[2].AsInt(0),       // x
      args[3].AsInt(0),       // y
      env
    );
  }
};

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KShowStatic", "cc", KShowStatic::Create, 0);

  env->AddFunction("KFMCycleAnalyze", "cc[lscale]f[costth]f[adj2224]f[adj30]f", KFMCycleAnalyze::Create, 0);
  env->AddFunction("Print", "cs[x]i[y]i", Print::Create, 0);
}

#define NOMINMAX
#include <Windows.h>

void AddFuncFMKernel(IScriptEnvironment* env);
void AddFuncMergeStatic(IScriptEnvironment* env);
void AddFuncCombingAnalyze(IScriptEnvironment* env);
void AddFuncDebandKernel(IScriptEnvironment* env);
void AddFuncUCF(IScriptEnvironment* env);

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
  AddFuncMergeStatic(env);
  AddFuncCombingAnalyze(env);
  AddFuncDebandKernel(env);
  AddFuncUCF(env);

  return "K Field Matching Plugin";
}
