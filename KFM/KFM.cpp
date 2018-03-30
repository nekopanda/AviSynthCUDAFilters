
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

PulldownPattern::PulldownPattern()
	: fields()
{
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
  , p30()
{
  const PulldownPattern* patterns[] = { &p2323, &p2233, &p30 };

  for (int p = 0; p < 3; ++p) {
    for (int i = 0; i < 9; ++i) {
      allpatterns[p * 9 + i] = patterns[p]->GetPattern(i);
    }
  }
}

Frame24Info PulldownPatterns::GetFrame24(int patternIndex, int n24) const {
  Frame24Info info;
  info.cycleIndex = n24 / 4;
  info.frameIndex = n24 % 4;

	int searchFrame = info.frameIndex;

	// パターンが30pの場合は、5枚中真ん中の1枚を消す
	// 30pの場合は、24pにした時点で5枚中1枚が失われてしまうので、正しく60pに復元することはできない
	// 30p部分は60pクリップから取得されるので基本的には問題ないが、
	// 前後のサイクルが24pで、サイクル境界の空きフレームとして30p部分も取得されることがある
	// なので、5枚中、最初と最後のフレームだけは正しく60pに復元する必要がある
	// 以下の処理がないと最後のフレーム(4枚目)がズレてしまう
	if (patternIndex >= 18) {
		if (searchFrame >= 2) ++searchFrame;
	}

  const PulldownPatternField* ptn = allpatterns[patternIndex];
  int fldstart = 0;
  int nframes = 0;
  for (int i = 0; i < 14; ++i) {
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
  Frame24Info info;
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

std::pair<int, float> PulldownPatterns::Matching(const FMData* data, int width, int height, float costth) const
{
  const PulldownPattern* patterns[] = { &p2323, &p2233, &p30 };

	std::vector<float> mtshima(9 * 3);
	std::vector<float> mtshimacost(9 * 3);

  // 各スコアを計算
  for (int p = 0; p < 3; ++p) {
    for (int i = 0; i < 9; ++i) {
      auto pattern = patterns[p]->GetPattern(i);
			mtshima[p * 9 + i] = RSplitScore(pattern, data->mftr);
			mtshimacost[p * 9 + i] = RSplitCost(pattern, data->mftr, data->mftcost, costth);
    }
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
public:
  KFMCycleAnalyze(PClip fmframe, PClip source, float lscale, float costth, IScriptEnvironment* env)
		: GenericVideoFilter(fmframe)
		, source(source)
    , srcvi(source->GetVideoInfo())
		, lscale(lscale)
		, costth(costth)
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
			PVideoFrame frame = child->GetFrame(cycle * 5 + i, env);
			memcpy(fmcnt + (i + 2) * 2, frame->GetReadPtr(), sizeof(fmcnt[0]) * 2);
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

		auto result = patterns.Matching(&data, srcvi.width, srcvi.height, costth);

    PVideoFrame dst = env->NewVideoFrame(vi);
    uint8_t* dstp = dst->GetWritePtr();
    memcpy(dstp, &result, sizeof(result));

		// フレームをCUDAに持っていった後、
    // CPUからも取得できるようにプロパティにも入れておく
    dst->SetProps("KFM_Pattern", result.first);
    dst->SetProps("KFM_Cost", result.second);

    return dst;
	}

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMCycleAnalyze(
      args[0].AsClip(),       // fmframe
      args[1].AsClip(),       // source
			(float)args[2].AsFloat(5.0f), // lscale
			(float)args[3].AsFloat(1.0f), // costth
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

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KShowStatic", "cc", KShowStatic::Create, 0);

  env->AddFunction("KFMCycleAnalyze", "cc[lscale]f[costth]f", KFMCycleAnalyze::Create, 0);
  env->AddFunction("KShowCombe", "c", KShowCombe::Create, 0);
}

#define NOMINMAX
#include <Windows.h>

void AddFuncFMKernel(IScriptEnvironment* env);
void AddFuncDebandKernel(IScriptEnvironment* env);

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
	AddFuncDebandKernel(env);

  return "K Field Matching Plugin";
}