
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>

#include "CommonFunctions.h"
#include "Overlap.hpp"
#include "TextOut.h"

template <typename pixel_t>
void Copy(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[x + y * dst_pitch] = src[x + y * src_pitch];
		}
	}
}

template <typename pixel_t>
void Average(pixel_t* dst, int dst_pitch, const pixel_t* src1, const pixel_t* src2, int src_pitch, int width, int height)
{
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[x + y * dst_pitch] = 
				(src1[x + y * src_pitch] + src2[x + y * src_pitch] + 1) >> 1;
		}
	}
}

template <typename pixel_t>
void VerticalClean(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height, int thresh)
{
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 0; x < width; ++x) {
      int p0 = src[x + (y - 1) * src_pitch];
      int p1 = src[x + (y + 0) * src_pitch];
      int p2 = src[x + (y + 1) * src_pitch];
      p0 = (std::abs(p0 - p1) <= thresh) ? p0 : p1;
      p2 = (std::abs(p2 - p1) <= thresh) ? p2 : p1;
      // 2項分布でマージ
      dst[x + y * dst_pitch] = (p0 + 2 * p1 + p2 + 2) >> 2;
    }
  }
}

struct KFMParam
{
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x4AE3B19D,
  };

  int magicKey;
  int version;

  bool tff;
  int width;
  int height;
  int pixelShift;
  int logUVx;
  int logUVy;

  int pixelType;
  int bitsPerPixel;

  int blkSize;
  int numBlkX;
  int numBlkY;

  float chromaScale;

  static const KFMParam* GetParam(const VideoInfo& vi, IScriptEnvironment* env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const KFMParam* param = (const KFMParam*)(void*)vi.num_audio_samples;
    if (param->magicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, KFMParam* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

static const int tbl2323[][2] = {
	{ 0, 0 },
	{ -1, 0 },
	{ 0, 1 },
	{ -1, 1 },
	{ 1, 0 }
};

static const int tbl2233[][2] = {
	{ 0, 0 },
	{ -1, 0 },
	{ 0, 1 },
	{ -1, 1 },
	{ 0, 2 },
	{ -1, 2 },
	{ 1, 3 },
	{ 0, 3 },
	{ -1, 3 },
	{ 1, 0 },
};

struct FieldMathingScore {
	float n1; // 1フィールド先
	float n2; // 2フィールド先

	FieldMathingScore operator+(const FieldMathingScore& o) {
		FieldMathingScore ret = { n1 + o.n1, n2 + o.n2 };
		return ret;
	}
	FieldMathingScore operator*(float o) {
		FieldMathingScore ret = { n1 * o, n2 * o };
		return ret;
	}
};

class KFMCoreBase
{
public:
  virtual ~KFMCoreBase() {}
  virtual void FieldToFrame(const PVideoFrame& src, const PVideoFrame& dst, bool isTop) = 0;
  virtual void SmoothField(PVideoFrame& frame, IScriptEnvironment* env) = 0;
  virtual void CompareFieldN1(const PVideoFrame& base, const PVideoFrame& ref, bool isTop, FieldMathingScore* fms) = 0;
  virtual void CompareFieldN2(const PVideoFrame& base, const PVideoFrame& ref, bool isTop, FieldMathingScore* fms) = 0;
};

template <typename pixel_t>
class KFMCore : public KFMCoreBase
{
  const KFMParam* prm;

public:
  KFMCore(const KFMParam* prm) : prm(prm) { }

	void FieldToFrame(const PVideoFrame& src, const PVideoFrame& dst, bool isTop)
	{
		const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
		const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
		const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = src->GetPitch(PLANAR_Y) >> prm->pixelShift;
		int pitchUV = src->GetPitch(PLANAR_U) >> prm->pixelShift;
		int widthUV = prm->width >> prm->logUVx;
		int heightUV = prm->height >> prm->logUVy;

		if (isTop == false) {
			srcY += pitchY;
			srcU += pitchUV;
			srcV += pitchUV;
		}

		// top field
		Copy<pixel_t>(dstY, pitchY * 2, srcY, pitchY * 2, prm->width, prm->height / 2);
		Copy<pixel_t>(dstU, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);

		// bottom field
		Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcY, pitchY * 2, prm->width, prm->height / 2);
		Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);
	}

  // フィールドごとに平滑化フィルタを掛ける
  void SmoothField(PVideoFrame& frame, IScriptEnvironment* env)
  {
    env->MakeWritable(&frame);

    pixel_t* dstY = reinterpret_cast<pixel_t*>(frame->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(frame->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(frame->GetWritePtr(PLANAR_V));

    int pitchY = frame->GetPitch(PLANAR_Y) >> prm->pixelShift;
    int pitchUV = frame->GetPitch(PLANAR_U) >> prm->pixelShift;
    int widthUV = prm->width >> prm->logUVx;
    int heightUV = prm->height >> prm->logUVy;

    std::unique_ptr<pixel_t[]> buf = std::unique_ptr<pixel_t[]>(new pixel_t[prm->width * 3]);
    pixel_t* bufY = buf.get();
    pixel_t* bufU = &bufY[prm->width];
    pixel_t* bufV = &bufY[prm->width];

    for (int y = 0; y < prm->height; ++y) {
      memcpy(bufY, &dstY[y * pitchY], sizeof(pixel_t) * pitchY);
      for (int x = 1; x < prm->width - 2; ++x) {
        dstY[x + y * pitchY] = (bufY[x - 1] + bufY[x] + bufY[x + 1] + bufY[x + 2]) >> 2;
      }
    }
    for (int y = 0; y < heightUV; ++y) {
      memcpy(bufU, &dstU[y * pitchUV], sizeof(pixel_t) * pitchUV);
      memcpy(bufV, &dstV[y * pitchUV], sizeof(pixel_t) * pitchUV);
      for (int x = 1; x < widthUV - 2; ++x) {
        dstU[x + y * pitchUV] = (bufU[x - 1] + bufU[x] + bufU[x + 1] + bufU[x + 2]) >> 2;
        dstV[x + y * pitchUV] = (bufV[x - 1] + bufV[x] + bufV[x + 1] + bufV[x + 2]) >> 2;
      }
    }
  }

	void CompareFieldN1(const PVideoFrame& base, const PVideoFrame& ref, bool isTop, FieldMathingScore* fms)
	{
		const pixel_t* baseY = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_Y));
		const pixel_t* baseU = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_U));
		const pixel_t* baseV = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_V));
		const pixel_t* refY = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_Y));
		const pixel_t* refU = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_U));
		const pixel_t* refV = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_V));

		int pitchY = base->GetPitch(PLANAR_Y) >> prm->pixelShift;
		int pitchUV = base->GetPitch(PLANAR_U) >> prm->pixelShift;
		int widthUV = prm->width >> prm->logUVx;
		int heightUV = prm->height >> prm->logUVy;

		for (int by = 0; by < prm->numBlkY; ++by) {
			for (int bx = 0; bx < prm->numBlkX; ++bx) {
				int yStart = by * prm->blkSize;
				int yEnd = min(yStart + prm->blkSize, prm->height);
				int xStart = bx * prm->blkSize;
				int xEnd = min(xStart + prm->blkSize, prm->width);

				float sumY = 0;

				for (int y = yStart + (isTop ? 0 : 1); y < yEnd; y += 2) {
          int ypp = max(y - 2, 0);
          int yp = max(y - 1, 0);
          int yn = min(y + 1, prm->height - 1);
          int ynn = min(y + 2, prm->height - 1);

					for (int x = xStart; x < xEnd; ++x) {
            pixel_t a = baseY[x + ypp * pitchY];
            pixel_t b = refY[x + yp * pitchY];
            pixel_t c = baseY[x + y * pitchY];
            pixel_t d = refY[x + yn * pitchY];
						pixel_t e = baseY[x + ynn * pitchY];
            float t= (a + 4 * c + e - 3 * (b + d)) * 0.1667f;
            t *= t;
						if(t > 15*15) {
							t = t * 16 - 15*15*15;
						}
						sumY += t;
					}
				}

				int yStartUV = yStart >> prm->logUVy;
				int yEndUV = yEnd >> prm->logUVy;
				int xStartUV = xStart >> prm->logUVx;
				int xEndUV = xEnd >> prm->logUVx;

				float sumUV = 0;

				for (int y = yStartUV + (isTop ? 0 : 1); y < yEndUV; y += 2) {
          int ypp = max(y - 2, 0);
          int yp = max(y - 1, 0);
          int yn = min(y + 1, heightUV - 1);
          int ynn = min(y + 2, heightUV - 1);

					for (int x = xStartUV; x < xEndUV; ++x) {
						{
              pixel_t a = baseU[x + ypp * pitchUV];
              pixel_t b = refU[x + yp * pitchUV];
              pixel_t c = baseU[x + y * pitchUV];
              pixel_t d = refU[x + yn * pitchUV];
              pixel_t e = baseU[x + ynn * pitchUV];
              float t = (a + 4 * c + e - 3 * (b + d)) * 0.1667f;
              t *= t;
              if (t > 15 * 15) {
                t = t * 16 - 15 * 15 * 15;
              }
							sumUV += t;
						}
						{
              pixel_t a = baseV[x + ypp * pitchUV];
              pixel_t b = refV[x + yp * pitchUV];
              pixel_t c = baseV[x + y * pitchUV];
              pixel_t d = refV[x + yn * pitchUV];
              pixel_t e = baseV[x + ynn * pitchUV];
              float t = (a + 4 * c + e - 3 * (b + d)) * 0.1667f;
              t *= t;
              if (t > 15 * 15) {
                t = t * 16 - 15 * 15 * 15;
              }
							sumUV += t;
						}
					}
				}

				float sum = (sumY + sumUV * prm->chromaScale) * (1.0f/16.0f);

				// 1ピクセル単位にする
				sum *= 1.0f / ((xEnd - xStart) * (yEnd - yStart));

				fms[bx + by * prm->numBlkX].n1 = sum;
			}
		}
	}

	void CompareFieldN2(const PVideoFrame& base, const PVideoFrame& ref, bool isTop, FieldMathingScore* fms)
	{
		const pixel_t* baseY = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_Y));
		const pixel_t* baseU = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_U));
		const pixel_t* baseV = reinterpret_cast<const pixel_t*>(base->GetReadPtr(PLANAR_V));
		const pixel_t* refY = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_Y));
		const pixel_t* refU = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_U));
		const pixel_t* refV = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(PLANAR_V));

		int pitchY = base->GetPitch(PLANAR_Y) >> prm->pixelShift;
		int pitchUV = base->GetPitch(PLANAR_U) >> prm->pixelShift;
		int widthUV = prm->width >> prm->logUVx;
		int heightUV = prm->height >> prm->logUVy;

		for (int by = 0; by < prm->numBlkY; ++by) {
			for (int bx = 0; bx < prm->numBlkX; ++bx) {
				int yStart = by * prm->blkSize;
				int yEnd = min(yStart + prm->blkSize, prm->height);
				int xStart = bx * prm->blkSize;
				int xEnd = min(xStart + prm->blkSize, prm->width);

				float sumY = 0;

				for (int y = yStart + (isTop ? 0 : 1); y < yEnd; y += 2) {
					for (int x = xStart; x < xEnd; ++x) {
						pixel_t b = baseY[x + y * pitchY];
						pixel_t r = refY[x + y * pitchY];
						sumY += (r - b) * (r - b);
					}
				}

				int yStartUV = yStart >> prm->logUVy;
				int yEndUV = yEnd >> prm->logUVy;
				int xStartUV = xStart >> prm->logUVx;
				int xEndUV = xEnd >> prm->logUVx;

				float sumUV = 0;

				for (int y = yStartUV + (isTop ? 0 : 1); y < yEndUV; y += 2) {
					for (int x = xStartUV; x < xEndUV; ++x) {
						{
							pixel_t b = baseU[x + y * pitchUV];
							pixel_t r = refU[x + y * pitchUV];
							sumUV += (r - b) * (r - b);
						}
						{
							pixel_t b = baseV[x + y * pitchUV];
							pixel_t r = refV[x + y * pitchUV];
							sumUV += (r - b) * (r - b);
						}
					}
				}

				float sum = sumY + sumUV * prm->chromaScale;

				// 1ピクセル単位にする
				sum *= 1.0f / ((xEnd - xStart) * (yEnd - yStart));

				fms[bx + by * prm->numBlkX].n2 = sum;
			}
		}
	}
};

// パターンスコア
typedef float PSCORE;

struct KFMFrame {
  int pattern;
  float reliability; // 信頼度
  FieldMathingScore fms[1];
};

class TelecinePattern {
	bool limitUpper;

	FieldMathingScore MatchingPattern2323(const FieldMathingScore* pBlks, int pattern)
	{
		FieldMathingScore result = { 0 };
		if (pattern == 0) { // 2323
			result = result + MatchingScore(pBlks + 0, false);
			result = result + MatchingScore(pBlks + 2, true);
			result = result + MatchingScore(pBlks + 5, false);
			result = result + MatchingScore(pBlks + 7, true);
		}
		else { // 3232
			result = result + MatchingScore(pBlks + 0, true);
			result = result + MatchingScore(pBlks + 3, false);
			result = result + MatchingScore(pBlks + 5, true);
			result = result + MatchingScore(pBlks + 8, false);
		}
		return result;
	}

	FieldMathingScore MatchingPattern2233(const FieldMathingScore* pBlks, int pattern)
	{
		FieldMathingScore result = { 0 };
		switch (pattern) {
		case 0: // 2233
			result = result + MatchingScore(pBlks + 0, false);
			result = result + MatchingScore(pBlks + 2, false);
			result = result + MatchingScore(pBlks + 4, true);
			result = result + MatchingScore(pBlks + 7, true);
			break;
		case 1: // 2332
			result = result + MatchingScore(pBlks + 0, false);
			result = result + MatchingScore(pBlks + 2, true);
			result = result + MatchingScore(pBlks + 5, true);
			result = result + MatchingScore(pBlks + 8, false);
			break;
		case 2: // 3322
			result = result + MatchingScore(pBlks + 0, true);
			result = result + MatchingScore(pBlks + 3, true);
			result = result + MatchingScore(pBlks + 6, false);
			result = result + MatchingScore(pBlks + 8, false);
			break;
		case 3: // 3223
			result = result + MatchingScore(pBlks + 0, true);
			result = result + MatchingScore(pBlks + 3, false);
			result = result + MatchingScore(pBlks + 5, false);
			result = result + MatchingScore(pBlks + 7, true);
			break;
		}
		return result;
	}
public:
	TelecinePattern(bool limitUpper) : limitUpper(limitUpper) { }

	FieldMathingScore MatchingScore(const FieldMathingScore* pBlk, bool is3)
	{
		FieldMathingScore ret = { 0 };
		if (is3) {
			ret.n1 = (pBlk[0].n1 + pBlk[1].n1) / 4;
			ret.n2 = pBlk[0].n2 / 2;
		}
		else {
			ret.n1 = pBlk[0].n1;
		}
		if (limitUpper) {
			// 上限制限
			if (ret.n1 >= 10 * 10) {
				ret.n1 = 5 * 5 * 100;
			}
			else if (ret.n1 >= 3 * 3) {
				ret.n1 *= 25;
			}
			if (ret.n2 >= 5 * 5) {
				ret.n2 = 5 * 5 * 100;
			}
			else if (ret.n2 >= 3 * 3) {
				ret.n2 *= 25;
			}
		}
		return ret;
	}

	FieldMathingScore MatchingPattern(const FieldMathingScore* pBlks, int pattern)
	{
		FieldMathingScore s;
		if (pattern < 5) {
			int offset = tbl2323[pattern][0];
			s = MatchingPattern2323(&pBlks[offset + 2], tbl2323[pattern][1]);
			// 前が空いているパターンは前に３フィールド連続があるはずなのでそれもチェック
			if (offset == 1) {
				s = (s + MatchingScore(pBlks, true)) * 0.8f;
			}
			// 後ろが空いているパターンは後ろ２フィールドは確実に連続するはずなのでそれもチェック
			if (offset == -1) {
				s = (s + MatchingScore(pBlks + 11, false)) * 0.8f;
			}
		}
		else {
			int offset = tbl2233[pattern - 5][0];
			s = MatchingPattern2233(&pBlks[offset + 2], tbl2233[pattern - 5][1]);
			// 前が空いているパターンは前に３フィールド連続があるはずなのでそれもチェック
			if (offset == 1) {
				s = (s + MatchingScore(pBlks, true)) * 0.8f;
			}
			// 後ろが空いているパターンは後ろ２フィールドは確実に連続するはずなのでそれもチェック
			if (offset == -1) {
				s = (s + MatchingScore(pBlks + 11, false)) * 0.8f;
			}
		}
		return s;
	}
};


class KFM : public GenericVideoFilter
{
  KFMParam prm;

	VideoInfo srcvi;
  VideoInfo tmpvi;

  std::unique_ptr<KFMCoreBase> core;

  KFMCoreBase* CreateCore(IScriptEnvironment* env)
  {
    if (prm.pixelShift == 0) {
      return new KFMCore<uint8_t>(&prm);
    }
    else {
      return new KFMCore<uint16_t>(&prm);
    }
  }

	void CalcAllPatternScore(const FieldMathingScore* pBlks, PSCORE* score)
	{
		TelecinePattern tp(true);
		for (int i = 0; i < 15; ++i) {
			FieldMathingScore s = tp.MatchingPattern(pBlks, i);
			score[i] = s.n1 + s.n2;
		}
	}

public:

	KFM(PClip child, IScriptEnvironment* env)
		: GenericVideoFilter(child)
    , srcvi(vi)
    , tmpvi()
	{
    // prm生成
    prm.magicKey = KFMParam::MAGIC_KEY;
    prm.version = KFMParam::VERSION;
    prm.tff = child->GetParity(0);
    prm.width = srcvi.width;
    prm.height = srcvi.height;
    prm.pixelShift = (srcvi.ComponentSize() == 1) ? 0 : 1;
    prm.logUVx = srcvi.GetPlaneHeightSubsampling(PLANAR_U);
    prm.logUVy = srcvi.GetPlaneWidthSubsampling(PLANAR_U);
    prm.pixelType = srcvi.pixel_type;
    prm.bitsPerPixel = srcvi.BitsPerComponent();
    prm.blkSize = 16;
    prm.numBlkX = (srcvi.width + prm.blkSize - 1) / prm.blkSize;
    prm.numBlkY = (srcvi.height + prm.blkSize - 1) / prm.blkSize;
    prm.chromaScale = 1.0f;
    KFMParam::SetParam(vi, &prm);

    core = std::unique_ptr<KFMCoreBase>(CreateCore(env));

    // フレームレート
    vi.MulDivFPS(1, 5);
    vi.num_frames = nblocks(srcvi.num_frames, 5);

    // データサイズをセット
    int out_frame_bytes = sizeof(KFMFrame) + sizeof(FieldMathingScore) * prm.numBlkX * prm.numBlkY * 14;
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 2048;
    vi.height = nblocks(out_frame_bytes, vi.width * 4);

    // メモリを確保するデバイスとかマルチスレッドとかメモリ再利用とか考えたくないので、
    // ワークメモリの確保も全部都度NewVideoFrameで行う
    int tmp_frame_bytes = sizeof(PSCORE) * prm.numBlkX * prm.numBlkY * 15;
    tmpvi.pixel_type = VideoInfo::CS_BGR32;
    tmpvi.width = 2048;
    tmpvi.height = nblocks(tmp_frame_bytes, vi.width * 4);
	}

  PVideoFrame __stdcall GetFrame(int cycleNumber, IScriptEnvironment* env)
	{
		// 必要なフレームを揃える
		PVideoFrame frames[7];
		for (int i = 0; i < 7; ++i) {
			int fn = cycleNumber * 5 + i - 1;
			if (fn < 0) {
				frames[i] = env->NewVideoFrame(srcvi);
				core->FieldToFrame(child->GetFrame(0, env), frames[i], prm.tff ? true : false);
			}
			else if (fn >= srcvi.num_frames) {
				frames[i] = env->NewVideoFrame(srcvi);
        core->FieldToFrame(child->GetFrame(srcvi.num_frames - 1, env), frames[i], prm.tff ? false : true);
			}
			else {
				frames[i] = child->GetFrame(fn, env);
			}
      core->SmoothField(frames[i], env);
		}

    // メモリ確保
    PVideoFrame outframe = env->NewVideoFrame(vi);
    PVideoFrame tmpframe = env->NewVideoFrame(tmpvi);

    KFMFrame *fmframe = (KFMFrame*)outframe->GetWritePtr();

		// ブロックごとのマッチング計算
    FieldMathingScore *fms = fmframe->fms;
		int fmsPitch = prm.numBlkX * prm.numBlkY;
		for (int i = 0; i < 7; ++i) {
      core->CompareFieldN1(frames[i], frames[i], prm.tff, &fms[(i * 2 + 0) * fmsPitch]);
			if (i < 6) {
        core->CompareFieldN2(frames[i], frames[i + 1], prm.tff, &fms[(i * 2 + 0) * fmsPitch]);
        core->CompareFieldN1(frames[i], frames[i + 1], !prm.tff, &fms[(i * 2 + 1) * fmsPitch]);
        core->CompareFieldN2(frames[i], frames[i + 1], !prm.tff, &fms[(i * 2 + 1) * fmsPitch]);
			}
		}

		// パターンとのマッチングを計算
		PSCORE *blockpms = (PSCORE*)tmpframe->GetWritePtr();
		for (int by = 0; by < prm.numBlkY; ++by) {
			for (int bx = 0; bx < prm.numBlkX; ++bx) {
				int blkidx = bx + by * prm.numBlkX;
				FieldMathingScore blks[14];
				for (int i = 0; i < 14; ++i) {
					blks[i] = fms[i * fmsPitch + blkidx];
				}
				CalcAllPatternScore(blks, &blockpms[15 * blkidx]);
			}
		}

		// パターンごとに合計
    PSCORE pms[15] = { 0 };
		for (int by = 0; by < prm.numBlkY; ++by) {
			for (int bx = 0; bx < prm.numBlkX; ++bx) {
				int blkidx = bx + by * prm.numBlkX;
				for (int i = 0; i < 15; ++i) {
					pms[i] += blockpms[15 * blkidx + i];
				}
			}
		}
		PSCORE sum = 0;
    // 見やすくするために値を小さくする
    for (int i = 0; i < 15; ++i) {
      pms[i] /= fmsPitch;
			sum += pms[i];
    }

		// 最良パターン
		int pattern = 0;
		PSCORE curscore = pms[0];
		for (int i = 1; i < 15; ++i) {
			if (curscore > pms[i]) {
				curscore = pms[i];
				pattern = i;
			}
		}

    fmframe->pattern = pattern;
    fmframe->reliability = (sum / 15) / curscore;

    return outframe;
	}

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFM(
      args[0].AsClip(),       // clip
      env
    );
  }
};

class KIVTCCoreBase
{
public:
  virtual ~KIVTCCoreBase() { }
  virtual void CreateWeaveFrame2F(const PVideoFrame& srct, const PVideoFrame& srcb, const PVideoFrame& dst) = 0;
  virtual void CreateWeaveFrame3F(const PVideoFrame& src, const PVideoFrame& rf, int rfIndex, const PVideoFrame& dst) = 0;
};

template <typename pixel_t>
class KIVTCCore : public KIVTCCoreBase
{
  const KFMParam* prm;

public:
  KIVTCCore(const KFMParam* prm) : prm(prm) { }

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

    int pitchY = srct->GetPitch(PLANAR_Y) >> prm->pixelShift;
    int pitchUV = srct->GetPitch(PLANAR_U) >> prm->pixelShift;
    int widthUV = prm->width >> prm->logUVx;
    int heightUV = prm->height >> prm->logUVy;

    // copy top
    Copy<pixel_t>(dstY, pitchY * 2, srctY, pitchY * 2, prm->width, prm->height / 2);
    Copy<pixel_t>(dstU, pitchUV * 2, srctU, pitchUV * 2, widthUV, heightUV / 2);
    Copy<pixel_t>(dstV, pitchUV * 2, srctV, pitchUV * 2, widthUV, heightUV / 2);

    // copy bottom
    Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcbY + pitchY, pitchY * 2, prm->width, prm->height / 2);
    Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcbU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
    Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcbV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
  }

  // rfIndex:  0:top, 1:bottom
  void CreateWeaveFrame3F(const PVideoFrame& src, const PVideoFrame& rf, int rfIndex, const PVideoFrame& dst)
  {
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    const pixel_t* rfY = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_Y));
    const pixel_t* rfU = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_U));
    const pixel_t* rfV = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) >> prm->pixelShift;
    int pitchUV = src->GetPitch(PLANAR_U) >> prm->pixelShift;
    int widthUV = prm->width >> prm->logUVx;
    int heightUV = prm->height >> prm->logUVy;

    if (rfIndex == 0) {
      // average top
      Average<pixel_t>(dstY, pitchY * 2, srcY, rfY, pitchY * 2, prm->width, prm->height / 2);
      Average<pixel_t>(dstU, pitchUV * 2, srcU, rfU, pitchUV * 2, widthUV, heightUV / 2);
      Average<pixel_t>(dstV, pitchUV * 2, srcV, rfV, pitchUV * 2, widthUV, heightUV / 2);
      // copy bottom
      Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, pitchY * 2, prm->width, prm->height / 2);
      Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
      Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
    }
    else {
      // copy top
      Copy<pixel_t>(dstY, pitchY * 2, srcY, pitchY * 2, prm->width, prm->height / 2);
      Copy<pixel_t>(dstU, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
      Copy<pixel_t>(dstV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);
      // average bottom
      Average<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, rfY + pitchY, pitchY * 2, prm->width, prm->height / 2);
      Average<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, rfU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
      Average<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, rfV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
    }
  }
};

class KIVTC : public GenericVideoFilter
{
  PClip fmclip;

  const KFMParam* prm;

  std::unique_ptr<KIVTCCoreBase> core;

  KIVTCCoreBase* CreateCore(IScriptEnvironment* env)
  {
    if (prm->pixelShift == 0) {
      return new KIVTCCore<uint8_t>(prm);
    }
    else {
      return new KIVTCCore<uint16_t>(prm);
    }
  }

  PVideoFrame CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, IScriptEnvironment* env)
  {
    // fstartは0or1にする
    if (fstart < 0 || fstart >= 2) {
      n += fstart / 2;
      fstart &= 1;
    }

    assert(fstart == 0 || fstart == 1);
    assert(fnum == 2 || fnum == 3);

    if (fstart == 0 && fnum == 2) {
      return clip->GetFrame(n, env);
    }
    else {
      PVideoFrame cur = clip->GetFrame(n, env);
      PVideoFrame nxt = clip->GetFrame(n + 1, env);
      PVideoFrame dst = env->NewVideoFrame(vi);
      if (fstart == 0 && fnum == 3) {
        core->CreateWeaveFrame3F(cur, nxt, !prm->tff, dst);
      }
      else if (fstart == 1 && fnum == 3) {
        core->CreateWeaveFrame3F(nxt, cur, prm->tff, dst);
      }
      else if (fstart == 1 && fnum == 2) {
        if (prm->tff) {
          core->CreateWeaveFrame2F(nxt, cur, dst);
        }
        else {
          core->CreateWeaveFrame2F(cur, nxt, dst);
        }
      }
      return dst;
    }
  }

  void DrawInfo(PVideoFrame& dst, const KFMFrame* fmframe, IScriptEnvironment* env) {
    env->MakeWritable(&dst);

    char buf[100]; sprintf(buf, "KFM: %d (%.1f)", fmframe->pattern, fmframe->reliability);
    DrawText(dst, true, 0, 0, buf);
  }

public:
  KIVTC(PClip child, PClip fmclip, IScriptEnvironment* env)
    : GenericVideoFilter(child)
    , fmclip(fmclip)
    , prm(KFMParam::GetParam(fmclip->GetVideoInfo(), env))
  {
    core = std::unique_ptr<KIVTCCoreBase>(CreateCore(env));

    // フレームレート
    vi.MulDivFPS(4, 5);
    vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    int cycleIndex = n / 4;
    int frameIndex24 = n % 4;
    PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
    const KFMFrame* fmframe = (KFMFrame*)fm->GetReadPtr();
    int pattern = fmframe->pattern;

    int fstart;
    int fnum;

    if (pattern < 5) {
      int offsets[] = { 0, 2, 5, 7, 10, 12, 15, 17, 20 };
      int idx24 = tbl2323[pattern][1];
      fstart = cycleIndex * 10 + tbl2323[pattern][0] +
        (offsets[frameIndex24 + idx24] - offsets[idx24]);
      fnum = offsets[frameIndex24 + idx24 + 1] - offsets[frameIndex24 + idx24];
    }
    else {
      int offsets[] = { 0, 2, 4, 7, 10, 12, 14, 17, 20 };
      int idx24 = tbl2233[pattern - 5][1];
      fstart = cycleIndex * 10 + tbl2233[pattern - 5][0] +
        (offsets[frameIndex24 + idx24] - offsets[idx24]);
      fnum = offsets[frameIndex24 + idx24 + 1] - offsets[frameIndex24 + idx24];
    }

    PVideoFrame out = CreateWeaveFrame(child, 0, fstart, fnum, env);

    //DrawInfo(out, fmframe, env);

    return out;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KIVTC(
      args[0].AsClip(),       // clip
      args[1].AsClip(),       // fmclip
      env
    );
  }
};

class KTCMergeCoreBase
{
public:
  virtual ~KTCMergeCoreBase() { }
  virtual int WorkSize() const = 0;
  virtual void Merge(const bool* match,
    const PVideoFrame& src24, const PVideoFrame& src60, const PVideoFrame& dst,
    uint8_t* work) = 0;
};

template <typename pixel_t>
class KTCMergeCore : public KTCMergeCoreBase
{
  typedef typename std::conditional <sizeof(pixel_t) == 1, short, int>::type tmp_t;

  const KFMParam* prm;

  std::unique_ptr<OverlapWindows> wins;
  std::unique_ptr<OverlapWindows> winsUV;

  void ProcPlane(const bool* match, bool isUV,
    const pixel_t* src24, const pixel_t* src60, pixel_t* dst, int pitch, tmp_t* work)
  {
    OverlapWindows* pwin = isUV ? winsUV.get() : wins.get();

    int logx = isUV ? prm->logUVx : 0;
    int logy = isUV ? prm->logUVy : 0;

    int winSizeX = (prm->blkSize * 2) >> logx;
    int winSizeY = (prm->blkSize * 2) >> logy;
    int stepSizeX = (prm->blkSize) >> logx;
    int stepSizeY = (prm->blkSize) >> logy;
    int halfOvrX = (prm->blkSize / 2) >> logx;
    int halfOvrY = (prm->blkSize / 2) >> logy;
    int width = prm->width >> logx;
    int height = prm->height >> logy;

    // ゼロ初期化
    for (int y = 0; y<height; y++) {
      for (int x = 0; x<width; x++) {
        work[x + y * width] = 0;
      }
    }

    // workに足し合わせる
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int xwstart = bx * stepSizeX - halfOvrX;
        int xwend = xwstart + winSizeX;
        int ywstart = by * stepSizeY - halfOvrY;
        int ywend = ywstart + winSizeY;

        const pixel_t* src = match[bx + by * prm->numBlkX] ? src24 : src60;
        const pixel_t* srcblk = src + xwstart + ywstart * pitch;
        tmp_t* dstblk = work + xwstart + ywstart * width;
        const short* win = pwin->GetWindow(bx, by, prm->numBlkX, prm->numBlkY);

        int xstart = (xwstart < 0) ? halfOvrX : 0;
        int xend = (xwend >= width) ? winSizeX - (xwend - width) : winSizeX;
        int ystart = (ywstart < 0) ? halfOvrY : 0;
        int yend = (ywend >= height) ? winSizeY - (ywend - height) : winSizeY;

        for (int y = ystart; y < yend; ++y) {
          for (int x = xstart; x < xend; ++x) {
            if (sizeof(pixel_t) == 1) {
              dstblk[x + y * width] += (srcblk[x + y * pitch] * win[x + y * winSizeX] + 256) >> 6;
            }
            else {
              dstblk[x + y * width] += srcblk[x + y * pitch] * win[x + y * winSizeX];
            }
          }
        }
      }
    }

    // dstに変換
    const int max_pixel_value = (1 << prm->bitsPerPixel) - 1;
    const int shift = sizeof(pixel_t) == 1 ? 5 : (5 + 6);
    for (int y = 0; y<height; y++) {
      for (int x = 0; x<width; x++) {
        int a = work[x + y * width] >> shift;
        dst[x + y * pitch] = min(max_pixel_value, a);
      }
    }
  }

public:
  KTCMergeCore(const KFMParam* prm)
    : prm(prm)
  {
    wins = std::unique_ptr<OverlapWindows>(
      new OverlapWindows(prm->blkSize * 2, prm->blkSize * 2,
        prm->blkSize, prm->blkSize));
    winsUV = std::unique_ptr<OverlapWindows>(
      new OverlapWindows((prm->blkSize * 2) >> prm->logUVx, (prm->blkSize * 2) >> prm->logUVy,
        prm->blkSize >> prm->logUVx, prm->blkSize >> prm->logUVy));
  }

  int WorkSize() const {
    return prm->width * prm->height * sizeof(tmp_t) * 3;
  }

  void Merge(const bool* match,
    const PVideoFrame& src24, const PVideoFrame& src60, const PVideoFrame& dst,
    uint8_t* work)
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

    int pitchY = src24->GetPitch(PLANAR_Y) >> prm->pixelShift;
    int pitchUV = src24->GetPitch(PLANAR_U) >> prm->pixelShift;

    tmp_t* workY = (tmp_t*)work;
    tmp_t* workU = &workY[prm->width * prm->height];
    tmp_t* workV = &workU[prm->width * prm->height];

    ProcPlane(match, false, src24Y, src60Y, dstY, pitchY, workY);
    ProcPlane(match, true, src24U, src60U, dstU, pitchUV, workU);
    ProcPlane(match, true, src24V, src60V, dstV, pitchUV, workV);
  }
};

struct Frame24Info {
	int index24;     // サイクル内での24pにおけるフレーム番号
	int start60;     // 24pフレームの開始フィールド番号
	bool is3;        // 24pフレームのフィールド数
};

class KTCMergeMatching : public GenericVideoFilter
{
	const KFMParam* prm;
	VideoInfo tmpvi;

	float thMatch;
	float thDiff;
	int tempWidth;
	int spatialWidth;

	void GetPatternMatch(const KFMFrame* fmframe, bool* match)
	{
		TelecinePattern tp(false);
		int fmsPitch = prm->numBlkX * prm->numBlkY;
		for (int by = 0; by < prm->numBlkY; ++by) {
			for (int bx = 0; bx < prm->numBlkX; ++bx) {
				int blkidx = bx + by * prm->numBlkX;
				FieldMathingScore blks[14];
				for (int i = 0; i < 14; ++i) {
					blks[i] = fmframe->fms[i * fmsPitch + blkidx];
				}
				FieldMathingScore s = tp.MatchingPattern(blks, fmframe->pattern);

				bool ismatch;
				if (s.n2 > thDiff) {
					ismatch = false;
				}
				else {
					ismatch = (fmframe->reliability >= 2.5) || (s.n1 <= thMatch);
				}

				match[blkidx] = ismatch;
			}
		}
	}

	void GetFrameDiff(const KFMFrame* fmframe, bool* match)
	{
		int fmsPitch = prm->numBlkX * prm->numBlkY;
		for (int by = 0; by < prm->numBlkY; ++by) {
			for (int bx = 0; bx < prm->numBlkX; ++bx) {
				int blkidx = bx + by * prm->numBlkX;
				bool ismatch = true;
				// サイクル内のマッチングのみを見る
				for (int i = 2; i < 10; i += 2) {
					float f0 = fmframe->fms[i * fmsPitch + blkidx].n2;
					float f1 = fmframe->fms[(i + 1) * fmsPitch + blkidx].n2;
					ismatch &= ((f0 + f1) <= thDiff);
				}
				match[blkidx] |= ismatch;
			}
		}
	}

	void GetMatchFrame(int n, bool* match, IScriptEnvironment* env)
	{
		n = clamp(n, 0, vi.num_frames - 1);
		PVideoFrame fm = child->GetFrame(n, env);
		const KFMFrame* fmframe = (KFMFrame*)fm->GetReadPtr();

		GetPatternMatch(fmframe, match);

		// 文字とかの単フレームで見るとコーミングかどうか
		// 判断がつかないようなブロックに対する救済措置
		// フレームが前後のフレームと差がない場合はOKとみなす
		GetFrameDiff(fmframe, match);
	}

	void ExpandX(const bool* src, bool* dst, int numBlkX, int numBlkY) {
		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int xstart = std::max(0, bx - spatialWidth / 2);
				int xend = std::min(numBlkX - 1, bx + spatialWidth / 2 + 1);
				bool s = true;
				for (int x = xstart; x < xend; ++x) {
					s &= src[x + by * numBlkX];
				}
				dst[bx + by * numBlkX] = s;
			}
		}
	}

	void ExpandY(const bool* src, bool* dst, int numBlkX, int numBlkY) {
		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int ystart = std::max(0, by - spatialWidth / 2);
				int yend = std::min(numBlkY - 1, by + spatialWidth / 2 + 1);
				bool s = true;
				for (int y = ystart; y < yend; ++y) {
					s &= src[bx + y * numBlkX];
				}
				dst[bx + by * numBlkX] = s;
			}
		}
	}

public:
	KTCMergeMatching(PClip fmclip, float thMatch, float thDiff, int tempWidth, int spatialWidth, IScriptEnvironment* env)
		: GenericVideoFilter(fmclip)
		, prm(KFMParam::GetParam(vi, env))
		, tmpvi()
		, thMatch(thMatch)
		, thDiff(thDiff)
		, tempWidth(tempWidth)
		, spatialWidth(spatialWidth)
	{
		int numBlks = prm->numBlkX * prm->numBlkY;

		int tmp_frame_bytes = sizeof(bool) * numBlks;
		tmpvi.pixel_type = VideoInfo::CS_BGR32;
		tmpvi.width = 2048;
		tmpvi.height = nblocks(tmp_frame_bytes, vi.width * 4);

		int out_frame_bytes = sizeof(bool) * numBlks;
		vi.pixel_type = VideoInfo::CS_BGR32;
		vi.width = 2048;
		vi.height = nblocks(out_frame_bytes, vi.width * 4);
	}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		int numBlks = prm->numBlkX * prm->numBlkY;

		// メモリ確保
		PVideoFrame outframe = env->NewVideoFrame(vi);
		PVideoFrame tmpframe = env->NewVideoFrame(tmpvi);
		bool* match = (bool*)outframe->GetWritePtr();
		bool* work = (bool*)tmpframe->GetWritePtr();

		for (int by = 0; by < prm->numBlkY; ++by) {
			for (int bx = 0; bx < prm->numBlkX; ++bx) {
				match[bx + by * prm->numBlkX] = true;
			}
		}

		for (int i = 0; i < tempWidth; ++i) {
			GetMatchFrame(n + i - tempWidth / 2, work, env);

			for (int by = 0; by < prm->numBlkY; ++by) {
				for (int bx = 0; bx < prm->numBlkX; ++bx) {
					match[bx + by * prm->numBlkX] &= work[bx + by * prm->numBlkX];
				}
			}
		}

		ExpandX(match, work, prm->numBlkX, prm->numBlkY);
		ExpandY(work, match, prm->numBlkX, prm->numBlkY);

		return outframe;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KTCMergeMatching(
			args[0].AsClip(),       // fmclip
      (float)args[1].AsFloat(),
			(float)args[2].AsFloat(),
			args[3].AsInt(),
			args[4].AsInt(),
			env
		);
	}
};

class KTCMerge : public GenericVideoFilter
{
  PClip fmclip;
	PClip matchclip;
  PClip clip24;

  const KFMParam* prm;

	float thMatch;
	float thDiff;

  VideoInfo vi24;
  VideoInfo tmpvi;

  std::unique_ptr<KTCMergeCoreBase> core;

  std::vector<Frame24Info> table24;
  std::vector<Frame24Info> table60;

  KTCMergeCoreBase* CreateCore(IScriptEnvironment* env)
  {
    if (prm->pixelShift == 0) {
      return new KTCMergeCore<uint8_t>(prm);
    }
    else {
      return new KTCMergeCore<uint16_t>(prm);
    }
  }

  void MakeTable()
  {
    // パターンとフレーム番号からFrame24Infoが取得できるテーブルを作る

    bool is3_2323[] = { false, true, false, true, false, true, false ,true };
    bool is3_2233[] = { false, false, true, true, false, false, true ,true };

    table24.resize(4 * 15);
    table60.resize(10 * 15);

    for (int p = 0; p < 15; ++p) {
      int idx;
      bool *is3_ptr;
      if (p < 5) {
        idx = tbl2323[p][0];
        is3_ptr = &is3_2323[tbl2323[p][1]];
      }
      else {
        idx = tbl2233[p-5][0];
        is3_ptr = &is3_2233[tbl2233[p-5][1]];
      }

      // 最初と最後だけ入れられない可能性があるのでデフォルト値を入れておく
      Frame24Info first = { -1,0,true };
      Frame24Info last = { 4,7,true };
      table60[0 + p * 10] = first;
      table60[9 + p * 10] = last;

      for (int idx24 = 0; idx24 < 4; ++idx24) {
        bool is3 = is3_ptr[idx24];
        int start60 = idx;
        Frame24Info info = { idx24,start60,is3 };
        for (int i = 0; i < (is3 ? 3 : 2); ++i) {
          if (i == 0) {
            table24[idx24 + p * 4] = info;
          }
          if (idx >= 0 && idx < 10) {
            table60[idx + p * 10] = info;
          }
          ++idx;
        }
      }
    }
  }

  Frame24Info GetFrame24Info(int pattern, int frameIndex60)
  {
    return table60[frameIndex60 + pattern * 10];
  }

	int GetPattern(int cycleIndex, IScriptEnvironment* env)
	{
		PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
		const KFMFrame* fmframe = (KFMFrame*)fm->GetReadPtr();
		return fmframe->pattern;
	}

  void DrawInfo(PVideoFrame& dst, int unmatched, IScriptEnvironment* env) {
    env->MakeWritable(&dst);

    char buf[100]; sprintf(buf, "KTCMerge: %d", unmatched);
    DrawText(dst, true, 0, 0, buf);
  }

public:
  KTCMerge(PClip clip60, PClip clip24, PClip fmclip, PClip matchclip, IScriptEnvironment* env)
    : GenericVideoFilter(clip60)
    , fmclip(fmclip)
		, matchclip(matchclip)
    , clip24(clip24)
    , prm(KFMParam::GetParam(fmclip->GetVideoInfo(), env))
		, thMatch(thMatch)
		, thDiff(thDiff)
    , vi24(clip24->GetVideoInfo())
    , tmpvi()
  {

    core = std::unique_ptr<KTCMergeCoreBase>(CreateCore(env));

    MakeTable();

    int numBlks = prm->numBlkX * prm->numBlkY;

    // メモリを確保するデバイスとかマルチスレッドとかメモリ再利用とか考えたくないので、
    // ワークメモリの確保も全部都度NewVideoFrameで行う
    int tmp_frame_bytes = core->WorkSize();
    tmpvi.pixel_type = VideoInfo::CS_BGR32;
    tmpvi.width = 2048;
    tmpvi.height = nblocks(tmp_frame_bytes, vi.width * 4);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    int numBlks = prm->numBlkX * prm->numBlkY;

    int cycleIndex = n / 10;
    int frameIndex60 = n % 10;

		PVideoFrame matchframe;
		matchframe = matchclip->GetFrame(cycleIndex, env);
		bool* match = (bool*)matchframe->GetReadPtr();

		int pattern = GetPattern(cycleIndex, env);
    Frame24Info frame24info = table60[frameIndex60 + pattern * 10];

		if (frame24info.index24 == 4) {
			if (cycleIndex + 1 >= fmclip->GetVideoInfo().num_frames) {
				// 後ろがないので前にしておく
				frame24info.index24 = 3;
			}
			else {
				// 後ろのパターンが前空きの場合は前にする
				bool isNoTop = false;
				int nextPattern = GetPattern(cycleIndex + 1, env);
				if (nextPattern < 5) {
					isNoTop = (tbl2323[nextPattern][0] == 1);
				}
				else {
					isNoTop = (tbl2233[nextPattern - 5][0] == 1);
				}
				if (isNoTop) {
					frame24info.index24 = 3;
				}
			}
		}

    int numUnmatchBlks = 0;
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int blkidx = bx + by * prm->numBlkX;
        if (match[blkidx] == false) {
          ++numUnmatchBlks;
        }
      }
    }

		// メモリ確保
		PVideoFrame outframe = env->NewVideoFrame(vi);
		PVideoFrame tmpframe = env->NewVideoFrame(tmpvi);
		uint8_t* work = (uint8_t*)tmpframe->GetWritePtr();

		int n24 = cycleIndex * 4 + frame24info.index24;

    if (numUnmatchBlks >= 0.35 * numBlks) {
      return child->GetFrame(n, env);
    }
    else if (numUnmatchBlks == 0) {
      return clip24->GetFrame(n24, env);
    }

    PVideoFrame src24 = clip24->GetFrame(n24, env);
    PVideoFrame src60 = child->GetFrame(n, env);

    core->Merge(match, src24, src60, outframe, work);

    //DrawInfo(outframe, numUnmatchBlks, env);

    return outframe;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
		AVSValue mathingArgs[5] = {
			args[2].AsClip(),       // fmclip
			args[3].AsFloat(50.0f),
			args[4].AsFloat(50.0f),
			args[5].AsInt(7),
			args[6].AsInt(5)
		};
		return new KTCMerge(
			args[0].AsClip(),       // clip60
			args[1].AsClip(),       // clip24
			args[2].AsClip(),       // fmclip
			env->Invoke("KTCMergeMatching", AVSValue(mathingArgs, 5)).AsClip(),
      env
    );
  }
};

class KMergeDev : public GenericVideoFilter
{
  enum {
    DIST = 1,
    N_REFS = DIST * 2 + 1,
    BLK_SIZE = 8
  };

  typedef uint8_t pixel_t;

  PClip source;

  float thresh;
  int debug;

  int logUVx;
  int logUVy;

  int nBlkX;
  int nBlkY;

  PVideoFrame GetRefFrame(int ref, IScriptEnvironment2* env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return source->GetFrame(ref, env);
  }
public:
  KMergeDev(PClip clip60, PClip source, float thresh, int debug, IScriptEnvironment* env)
    : GenericVideoFilter(clip60)
    , source(source)
    , thresh(thresh)
    , debug(debug)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    nBlkX = nblocks(vi.width, BLK_SIZE);
    nBlkY = nblocks(vi.height, BLK_SIZE);

    VideoInfo srcvi = source->GetVideoInfo();
    if (vi.num_frames != srcvi.num_frames * 2) {
      env->ThrowError("[KMergeDev] Num frames don't match");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    int n30 = n >> 1;

    PVideoFrame frames[N_REFS];
    for (int i = 0; i < N_REFS; ++i) {
      frames[i] = GetRefFrame(i + n30 - DIST, env);
    }
    PVideoFrame inframe = child->GetFrame(n, env);

    struct DIFF { float v[3]; };

    std::vector<DIFF> diffs(nBlkX * nBlkY);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    // diffを取得
    for (int p = 0; p < 3; ++p) {
      const uint8_t* pRefs[N_REFS];
      for (int i = 0; i < N_REFS; ++i) {
        pRefs[i] = frames[i]->GetReadPtr(planes[p]);
      }
      int pitch = frames[0]->GetPitch(planes[p]);

      int width = vi.width;
      int height = vi.height;
      int blksizeX = BLK_SIZE;
      int blksizeY = BLK_SIZE;
      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
        blksizeX >>= logUVx;
        blksizeY >>= logUVy;
      }

      for (int by = 0; by < nBlkY; ++by) {
        for (int bx = 0; bx < nBlkX; ++bx) {
          int ystart = by * blksizeY;
          int yend = std::min(ystart + blksizeY, height);
          int xstart = bx * blksizeX;
          int xend = std::min(xstart + blksizeX, width);

          int diff = 0;
          for (int y = ystart; y < yend; ++y) {
            for (int x = xstart; x < xend; ++x) {
              int off = x + y * pitch;
              int minv = 0xFFFF;
              int maxv = 0;
              for (int i = 0; i < N_REFS; ++i) {
                int v = pRefs[i][off];
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
              }
              int maxdiff = maxv - minv;
              diff += maxdiff;
            }
          }

          diffs[bx + by * nBlkX].v[p] = (float)diff / (blksizeX * blksizeY);
        }
      }
    }

    // フレーム作成
    PVideoFrame& src = frames[DIST];
    PVideoFrame dst = env->NewVideoFrame(vi);

    for (int p = 0; p < 3; ++p) {
      const pixel_t* pSrc = src->GetReadPtr(planes[p]);
      const pixel_t* pIn = inframe->GetReadPtr(planes[p]);
      pixel_t* pDst = dst->GetWritePtr(planes[p]);
      int pitch = src->GetPitch(planes[p]);

      int width = vi.width;
      int height = vi.height;
      int blksizeX = BLK_SIZE;
      int blksizeY = BLK_SIZE;
      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
        blksizeX >>= logUVx;
        blksizeY >>= logUVy;
      }

      for (int by = 0; by < nBlkY; ++by) {
        for (int bx = 0; bx < nBlkX; ++bx) {
          int ystart = by * blksizeY;
          int yend = std::min(ystart + blksizeY, height);
          int xstart = bx * blksizeX;
          int xend = std::min(xstart + blksizeX, width);

          DIFF& diff = diffs[bx + by * nBlkX];
          bool isStatic = (diff.v[0] < thresh) && (diff.v[1] < thresh) && (diff.v[2] < thresh);

          if ((debug == 1 && isStatic) || (debug == 2 && !isStatic)) {
            for (int y = ystart; y < yend; ++y) {
              for (int x = xstart; x < xend; ++x) {
                int off = x + y * pitch;
                pDst[off] = 20;
              }
            }
          }
          else {
            const pixel_t* pFrom = (isStatic ? pSrc : pIn);
            for (int y = ystart; y < yend; ++y) {
              for (int x = xstart; x < xend; ++x) {
                int off = x + y * pitch;
                pDst[off] = pFrom[off];
              }
            }
            if (isStatic) {
              // 上下方向の縞をなくすため上下のしきい値以下のピクセルと平均化する
              for (int y = ystart + 1; y < yend - 1; ++y) {
                for (int x = xstart; x < xend; ++x) {
                  int p0 = pFrom[x + (y - 1) * pitch];
                  int p1 = pFrom[x + (y + 0) * pitch];
                  int p2 = pFrom[x + (y + 1) * pitch];
                  p0 = (std::abs(p0 - p1) <= thresh) ? p0 : p1;
                  p2 = (std::abs(p2 - p1) <= thresh) ? p2 : p1;
                  // 2項分布でマージ
                  pDst[x + y * pitch] = (p0 + 2 * p1 + p2 + 2) >> 2;
                }
              }
            }
          }
        }
      }
    }

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KMergeDev(
      args[0].AsClip(),       // clip60
      args[1].AsClip(),       // source
      args[2].AsFloat(2),     // thresh
      args[3].AsInt(0),       // debug
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
	{ }

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame f0 = child->GetFrame(n, env);
		PVideoFrame f1 = child->GetFrame(n + 1, env);

    VideoInfo flagvi;
    if (vi.Is420()) flagvi.pixel_type = VideoInfo::CS_YV12;
    else if (vi.Is422()) flagvi.pixel_type = VideoInfo::CS_YV16;
    else if (vi.Is444()) flagvi.pixel_type = VideoInfo::CS_YV24;
    flagvi.width = vi.width;
    flagvi.height = vi.height;
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

class KFMAnalyzeFrame : public GenericVideoFilter
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
  KFMAnalyzeFrame(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
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
    return new KFMAnalyzeFrame(
      args[0].AsClip(),       // clip
      args[1].AsInt(10),       // threshMY
      args[2].AsInt(10),       // threshSY
      args[3].AsInt(10),       // threshMC
      args[4].AsInt(10),       // threshSC
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

template<int N> float CalcThreshold(const float* data)
{
  float buf[N];
  for (int i = 0; i < N; ++i) buf[i] = data[i];
  std::sort(buf, buf + N);
  float sum1 = buf[0];
  float sum2 = std::accumulate(buf + 1, buf + N, 0.0f);
  float v[N] = { 0 };
  for (int i = 1; i < N - 1; ++i) {
    float m1 = sum1 / i;
    float m2 = sum2 / (N - i);
    v[i] = i * (N - i) * (m1 - m2) * (m1 - m2);
    sum1 += buf[i];
    sum2 -= buf[i];
  }
  int maxidx = (int)(std::max_element(v + 1, v + N - 1) - v);
  return buf[maxidx];
}

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

float SplitCost(const PulldownPatternField* pattern, const float* fv, float& numshima) {
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
  numshima = sumnsplit;
  if (nsplitcoef == 0 && splitcoef == 0) {
    return 0;
  }
  return nsplitcoef / (splitcoef + 0.1f * nsplitcoef);
}

float MergeScore(const PulldownPatternField* pattern, const float* mergev, float& nummove) {
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
  nummove = splitcoef;
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

  Frame24InfoV2 GetFrame24(int patternIndex, int n24) const {
    Frame24InfoV2 info;
    info.cycleIndex = n24 / 4;
    info.frameIndex = n24 % 4;

    const PulldownPatternField* ptn = allpatterns[patternIndex];
    int fldstart = 0;
    int nframes = 0;
    for (int i = 0; i < 14; ++i) {
      if (ptn[i].split) {
        int nextfldstart = i + 1;
        int numflds = nextfldstart - fldstart;
        if ((nextfldstart - 2) * 2 >= numflds) {
          // 半分以上入ってればOK
          if (nframes++ == info.frameIndex) {
            info.fieldStartIndex = fldstart - 2;
            info.numFields = numflds;
            return info;
          }
        }
        fldstart = i + 1;
      }
    }
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
    std::vector<bool> shimaremain(9 * 3);

    std::vector<float> nummove(9 * 3);
    std::vector<float> numshima(9 * 3);
    std::vector<float> numlshima(9 * 3);

		// 各スコアを計算
		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < 9; ++i) {

				auto pattern = patterns[p]->GetPattern(i);
				shima[p * 9 + i] = SplitScore(pattern, data->fieldv, 0);
				shimabase[p * 9 + i] = SplitScore(pattern, data->fieldv, data->fieldbase);
				lshima[p * 9 + i] = SplitScore(pattern, data->fieldlv, 0);
				lshimabase[p * 9 + i] = SplitScore(pattern, data->fieldlv, data->fieldlbase);
				split[p * 9 + i] = SplitScore(pattern, data->splitv, 0);
        merge[p * 9 + i] = MergeScore(pattern, data->mergev, nummove[p * 9 + i]);

        shimacost[p * 9 + i] = SplitCost(pattern, data->fieldv, numshima[p * 9 + i]);
        lshimacost[p * 9 + i] = SplitCost(pattern, data->fieldlv, numlshima[p * 9 + i]);
        mergecost[p * 9 + i] = MergeScore(pattern, data->move, nummove[p * 9 + i]);

        numshima[p * 9 + i] *= 1.0f / (width * height);
        numlshima[p * 9 + i] *= 1.0f / (width * height);
        nummove[p * 9 + i] *= 1.0f / (width * height);
        // 1ピクセル当たり
        if (numlshima[p * 9 + i] > 0.001f || nummove[p * 9 + i] > 0.01f) {
          shimaremain[p * 9 + i] = true;
        }
        else {
          shimaremain[p * 9 + i] = false;
        }
			}
		}

    auto makeRet = [&](int n) {
      float cost = shimaremain[n] ? 10 : 0;
      cost += shimacost[n] + lshimacost[n] + mergecost[n];
      return std::pair<int, float>(n, numlshima[n] * 1000);
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
    int nummerge = std::count_if(data->mergev, data->mergev + 14, [](float f) { return f > 1.0f; });
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

class KFMCycleDev : public GenericVideoFilter
{
	PClip fmframe;
  VideoInfo srcvi;
	PulldownPatterns patterns;
public:
	KFMCycleDev(PClip source, PClip fmframe, IScriptEnvironment* env)
		: GenericVideoFilter(source)
		, fmframe(fmframe)
	{
    srcvi = vi;

    int out_bytes = sizeof(std::pair<int, float>);
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);
  }

	PVideoFrame __stdcall GetFrame(int cycle, IScriptEnvironment* env)
	{
		FMCount fmcnt[18];
		for (int i = -2; i <= 6; ++i) {
			PVideoFrame frame = fmframe->GetFrame(cycle * 5 + i, env);
			memcpy(fmcnt + (i + 2) * 2, frame->GetReadPtr(), sizeof(fmcnt[0]) * 2);
		}

		FMData data = { 0 };
		int fieldbase = INT_MAX;
		int fieldlbase = INT_MAX;
    for (int i = 0; i < 14; ++i) {
			data.fieldv[i] = fmcnt[i + 2].shima;
			data.fieldlv[i] = fmcnt[i + 2].lshima;
			data.move[i] = fmcnt[i + 2].move;
      data.splitv[i] = std::min(fmcnt[i + 1].move, fmcnt[i + 2].move);
			fieldbase = std::min(fieldbase, fmcnt[i + 2].shima);
			fieldlbase = std::min(fieldlbase, fmcnt[i + 2].lshima);

      if (fmcnt[i + 1].move > fmcnt[i + 2].move && fmcnt[i + 3].move > fmcnt[i + 2].move) {
        float sum = fmcnt[i + 1].move + fmcnt[i + 3].move;
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
    return new KFMCycleDev(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // fmclip
      env
    );
  }
};

class KTelecineCoreBase
{
public:
	virtual ~KTelecineCoreBase() { }
	virtual void CreateWeaveFrame2F(const PVideoFrame& srct, const PVideoFrame& srcb, const PVideoFrame& dst) = 0;
	virtual void CreateWeaveFrame3F(const PVideoFrame& src, const PVideoFrame& rf, int rfIndex, const PVideoFrame& dst) = 0;
  virtual void RemoveShima(const PVideoFrame& src, PVideoFrame& dst, int thresh) = 0;
  virtual void MakeSAD2F(const PVideoFrame& src, const PVideoFrame& rf, std::vector<int>& sads) = 0;
  virtual void MakeDevFrame(const PVideoFrame& src, const std::vector<int>& sads, int thresh, PVideoFrame& dst) = 0;
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

	// rfIndex:  0:top, 1:bottom
	void CreateWeaveFrame3F(const PVideoFrame& src, const PVideoFrame& rf, int rfIndex, const PVideoFrame& dst)
	{
		const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
		const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
		const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
		const pixel_t* rfY = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_Y));
		const pixel_t* rfU = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_U));
		const pixel_t* rfV = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_V));
		pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
		pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
		pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

		int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
		int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
		int widthUV = vi.width >> logUVx;
		int heightUV = vi.height >> logUVy;

		if (rfIndex == 0) {
			// average top
			Average<pixel_t>(dstY, pitchY * 2, srcY, rfY, pitchY * 2, vi.width, vi.height / 2);
			Average<pixel_t>(dstU, pitchUV * 2, srcU, rfU, pitchUV * 2, widthUV, heightUV / 2);
			Average<pixel_t>(dstV, pitchUV * 2, srcV, rfV, pitchUV * 2, widthUV, heightUV / 2);
			// copy bottom
			Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, pitchY * 2, vi.width, vi.height / 2);
			Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
			Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		}
		else {
			// copy top
			Copy<pixel_t>(dstY, pitchY * 2, srcY, pitchY * 2, vi.width, vi.height / 2);
			Copy<pixel_t>(dstU, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
			Copy<pixel_t>(dstV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);
			// average bottom
			Average<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, rfY + pitchY, pitchY * 2, vi.width, vi.height / 2);
			Average<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, rfU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
			Average<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, rfV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		}
	}

  void RemoveShima(const PVideoFrame& src, PVideoFrame& dst, int thresh)
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

    VerticalClean<pixel_t>(dstY, pitchY, srcY, pitchY, vi.width, vi.height, thresh);
    VerticalClean<pixel_t>(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV, thresh);
    VerticalClean<pixel_t>(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV, thresh);
  }

  void MakeSAD2F(const PVideoFrame& src, const PVideoFrame& rf, std::vector<int>& sads)
  {
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    const pixel_t* rfY = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_Y));
    const pixel_t* rfU = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_U));
    const pixel_t* rfV = reinterpret_cast<const pixel_t*>(rf->GetReadPtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;
    int blockSize = 16;
    int overlap = 8;
    int blockSizeUVx = blockSize >> logUVx;
    int blockSizeUVy = blockSize >> logUVy;
    int overlapUVx = overlap >> logUVx;
    int overlapUVy = overlap >> logUVy;
    int numBlkX = (vi.width - blockSize) / overlap + 1;
    int numBlkY = (vi.height - blockSize) / overlap + 1;
    
    sads.resize(numBlkX * numBlkY);

    for (int by = 0; by < numBlkY; ++by) {
      for (int bx = 0; bx < numBlkX; ++bx) {
        int yStart = by * overlap;
        int yEnd = yStart + blockSize;
        int xStart = bx * overlap;
        int xEnd = xStart + blockSize;
        int yStartUV = by * overlapUVy;
        int yEndUV = yStartUV + overlapUVy;
        int xStartUV = bx * overlapUVx;
        int xEndUV = xStartUV + overlapUVx;

        int sad = 0;
        for (int y = yStart; y < yEnd; ++y) {
          for (int x = xStart; x < xEnd; ++x) {
            int diff = srcY[x + y * pitchY] - rfY[x + y * pitchY];
            sad += (diff >= 0) ? diff : -diff;
          }
        }

        int sadC = 0;
        for (int y = yStartUV; y < yEndUV; ++y) {
          for (int x = xStartUV; x < xEndUV; ++x) {
            int diffU = srcU[x + y * pitchUV] - rfU[x + y * pitchUV];
            int diffV = srcV[x + y * pitchUV] - rfV[x + y * pitchUV];
            sadC += (diffU >= 0) ? diffU : -diffU;
            sadC += (diffV >= 0) ? diffV : -diffV;
          }
        }

        sads[bx + by * numBlkX] = sad + sadC;
      }
    }
  }

  void MakeDevFrame(const PVideoFrame& src, const std::vector<int>& sads, int thresh, PVideoFrame& dst)
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
    int blockSize = 16;
    int overlap = 8;
    int uniqueSize = blockSize - overlap;
    int blockSizeUVx = blockSize >> logUVx;
    int blockSizeUVy = blockSize >> logUVy;
    int overlapUVx = overlap >> logUVx;
    int overlapUVy = overlap >> logUVy;
    int uniqueSizeUVx = uniqueSize >> logUVx;
    int uniqueSizeUVy = uniqueSize >> logUVy;
    int numBlkX = (vi.width - blockSize) / overlap + 1;
    int numBlkY = (vi.height - blockSize) / overlap + 1;

    int blue[] = { 73, 230, 111 };

    for (int by = 0; by < numBlkY; ++by) {
      for (int bx = 0; bx < numBlkX; ++bx) {
        int yStart = by * overlap + overlap / 2;
        int yEnd = yStart + uniqueSize;
        int xStart = bx * overlap + overlap / 2;
        int xEnd = xStart + uniqueSize;
        int yStartUV = by * overlapUVy + overlapUVy / 2;
        int yEndUV = yStartUV + uniqueSizeUVy;
        int xStartUV = bx * overlapUVx + overlapUVx / 2;
        int xEndUV = xStartUV + uniqueSizeUVx;

        int sad = sads[bx + by * numBlkX];
        bool ok = (sad < thresh);

        for (int y = yStart; y < yEnd; ++y) {
          for (int x = xStart; x < xEnd; ++x) {
            dstY[x + y * pitchY] = ok ? srcY[x + y * pitchY] : blue[0];
          }
        }

        int sadC = 0;
        for (int y = yStartUV; y < yEndUV; ++y) {
          for (int x = xStartUV; x < xEndUV; ++x) {
            dstU[x + y * pitchUV] = ok ? srcU[x + y * pitchUV] : blue[1];
            dstV[x + y * pitchUV] = ok ? srcV[x + y * pitchUV] : blue[2];
          }
        }
      }
    }
  }
};

class KTelecine : public GenericVideoFilter
{
  PClip clip60;
	PClip fmclip;

  PulldownPatterns patterns;

  float thresh;
  float smooth;

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
			if (fstart == 0 && fnum >= 3) {
				core->CreateWeaveFrame3F(cur, nxt, !parity, dst);
			}
			else if (fstart == 1 && fnum >= 3) {
				core->CreateWeaveFrame3F(nxt, cur, parity, dst);
			}
			else if (fstart == 1 && fnum == 2) {
				if (parity) {
					core->CreateWeaveFrame2F(nxt, cur, dst);
				}
				else {
					core->CreateWeaveFrame2F(cur, nxt, dst);
				}
			}
			return dst;
		}
	}

  PVideoFrame CreateDiffFrame(PClip clip60, const PVideoFrame& src, int n, int fstart, int fnum, int parity, IScriptEnvironment* env)
  {
    fstart += n * 2;
    assert(fnum == 2 || fnum == 3 || fnum == 4);

    PVideoFrame s0 = clip60->GetFrame(fstart + 0, env);
    PVideoFrame s1 = clip60->GetFrame(fstart + 1, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    if (fnum == 2) {
      std::vector<int> sads;
      core->MakeSAD2F(s0, s1, sads);
      core->MakeDevFrame(src, sads, (int)(thresh * 16 * 16), dst);

      return dst;
    }
    else {
      return src;
    }
  }

	void DrawInfo(PVideoFrame& dst, const std::pair<int, float>* fmframe, int fnum, IScriptEnvironment* env) {
		env->MakeWritable(&dst);

		char buf[100]; sprintf(buf, "KFM: %d (%.1f) - %d", fmframe->first, fmframe->second, fnum);
		DrawText(dst, true, 0, 0, buf);
	}

public:
	KTelecine(PClip child, PClip clip60, PClip fmclip, float thresh, float smooth, IScriptEnvironment* env)
		: GenericVideoFilter(child)
    , clip60(clip60)
    , fmclip(fmclip)
    , thresh(thresh)
    , smooth(smooth)
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
    PVideoFrame rs = env->NewVideoFrame(vi); core->RemoveShima(out, rs, smooth);
    PVideoFrame out2 = CreateDiffFrame(clip60, rs, 0, fstart, frameInfo.numFields, parity, env);

		DrawInfo(out2, fmframe, frameInfo.numFields, env);

		return out2;
	}

	static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
	{
		return new KTelecine(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // clip60
			args[2].AsClip(),       // fmclip
      args[3].AsFloat(20),    // thresh
      args[4].AsFloat(20),    // thresh
			env
			);
	}
};

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KFM", "c", KFM::Create, 0);
  env->AddFunction("KIVTC", "cc", KIVTC::Create, 0);
	env->AddFunction("KTCMerge", "ccc[thmatch]f[thdiff]f[temp]i[spatial]i", KTCMerge::Create, 0);
	// 内部用
	env->AddFunction("KTCMergeMatching", "cffii", KTCMergeMatching::Create, 0);

  env->AddFunction("KMergeDev", "cc[thresh]f[debug]i", KMergeDev::Create, 0);
  env->AddFunction("KFMFrameDev", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameDev::Create, 0);
  env->AddFunction("KFMAnalyzeFrame", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMAnalyzeFrame::Create, 0);
  env->AddFunction("KFMCycleDev", "cc", KFMCycleDev::Create, 0);
  env->AddFunction("KTelecine", "ccc[thresh]f[smooth]f", KTelecine::Create, 0);
}

#include <Windows.h>

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

  return "K Field Matching Plugin";
}