
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
	int vbase;
	// 小さい縞の量
	float fieldv[14];
	float fieldbase[14];
	float fieldrv[14];
	// 大きい縞の量
	float fieldlv[14];
	float fieldlbase[14];
	float fieldrlv[14];
	// 動き量
	float move[14];
	// 動きから見た分離度
	float splitv[14];
	// 動きから見た3フィールド結合度
	float mergev[14];
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

float RSplitScore(const PulldownPatternField* pattern, const float* fv) {
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
	return splitcoef - nsplitcoef;
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

float RSplitCost(const PulldownPatternField* pattern, const float* rawfv, const float* fv, int vbase) {
	int nsplit = 0;
	float sumcost = 0;

	for (int i = 0; i < 14; ++i) {
		if (pattern[i].split) {
			nsplit++;
			if (fv[i] < 1.0f) {
				sumcost += (1.0f - fv[i]) * (rawfv[i] / vbase);
			}
		}
	}

	return sumcost / nsplit;
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

std::pair<int, float> PulldownPatterns::Matching(const FMData* data, int width, int height) const
{
  const PulldownPattern* patterns[] = { &p2323, &p2233, &p30 };

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

  //std::vector<float> mergecost(9 * 3);
  std::vector<float> shimacost(9 * 3);
  //std::vector<float> lshimacost(9 * 3);

  // 各スコアを計算
  for (int p = 0; p < 3; ++p) {
    for (int i = 0; i < 9; ++i) {

      auto pattern = patterns[p]->GetPattern(i);
      shima[p * 9 + i] = RSplitScore(pattern, data->fieldrv);
      shimabase[p * 9 + i] = SplitScore(pattern, data->fieldv, data->fieldbase);
      lshima[p * 9 + i] = RSplitScore(pattern, data->fieldrlv);
      lshimabase[p * 9 + i] = SplitScore(pattern, data->fieldlv, data->fieldlbase);
      split[p * 9 + i] = SplitScore(pattern, data->splitv, 0);

      shimacost[p * 9 + i] = RSplitCost(pattern, data->fieldv, data->fieldrv, data->vbase);
      //lshimacost[p * 9 + i] = RSplitCost(pattern, data->fieldlv, data->fieldrlv, data->vbase);

			if (PulldownPatterns::Is30p(p * 9 + i) == false) {
				merge[p * 9 + i] = MergeScore(pattern, data->mergev);
				//mergecost[p * 9 + i] = MergeScore(pattern, data->move);
			}
    }
  }

  auto makeRet = [&](int n) {
    //float cost = shimacost[n] + lshimacost[n] + mergecost[n];
		float cost = shimacost[n];
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

#pragma endregion

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


		FMData data = { 0 };
		data.vbase = (int)(srcvi.width * srcvi.height * 0.001f);
		int fieldbase = INT_MAX;
		int fieldlbase = INT_MAX;
    for (int i = 0; i < 14; ++i) {
			data.fieldv[i] = (float)fmcnt[i + 2].shima;
			data.fieldlv[i] = (float)fmcnt[i + 2].lshima;
			data.move[i] = (float)fmcnt[i + 2].move;
      data.splitv[i] = (float)std::min(fmcnt[i + 1].move, fmcnt[i + 2].move);
			fieldbase = std::min(fieldbase, fmcnt[i + 2].shima);
			fieldlbase = std::min(fieldlbase, fmcnt[i + 2].lshima);

			// relative
			data.fieldrv[i] = (fmcnt[i + 2].shima + data.vbase) * 2.0f / (fmcnt[i + 1].shima + fmcnt[i + 3].shima + data.vbase * 2.0f) - 1.0f;
			data.fieldrlv[i] = (fmcnt[i + 2].lshima + data.vbase) * 2.0f / (fmcnt[i + 1].lshima + fmcnt[i + 3].lshima + data.vbase * 2.0f) - 1.0f;

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

  env->AddFunction("KFMCycleAnalyze", "cc", KFMCycleAnalyze::Create, 0);
  env->AddFunction("KShowCombe", "c", KShowCombe::Create, 0);
}

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