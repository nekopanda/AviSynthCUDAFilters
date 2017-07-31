
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>

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

				// 上限制限
				if (sum >= 10 * 10) {
          sum = 5 * 5 * 100;
				}
        else if (sum >= 3 * 3) {
          sum *= 25;
        }

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

				// 上限制限
				if (sum >= 5 * 5) {
          sum = 5 * 5 * 100;
				}
        else if (sum >= 3 * 3) {
          sum *= 25;
        }

				fms[bx + by * prm->numBlkX].n2 = sum;
			}
		}
	}
};

// パターンスコア
typedef float PSCORE;

struct KFMFrame {
  int pattern;
  PSCORE score;
  FieldMathingScore fms[1];
};

PSCORE MatchingScore(const FieldMathingScore* pBlk, bool is3)
{
  PSCORE ret;
  if (is3) {
    ret = (pBlk[0].n1 + pBlk[0].n2 * 2 + pBlk[1].n1) / 4;
  }
  else {
    ret = pBlk[0].n1;
  }
  return ret;
}

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

	PSCORE MatchingPattern2323(const FieldMathingScore* pBlks, int pattern)
	{
    PSCORE result = 0;
		if (pattern == 0) { // 2323
			result += MatchingScore(pBlks + 0, false);
			result += MatchingScore(pBlks + 2, true);
			result += MatchingScore(pBlks + 5, false);
			result += MatchingScore(pBlks + 7, true);
		}
		else { // 3232
			result += MatchingScore(pBlks + 0, true);
			result += MatchingScore(pBlks + 3, false);
			result += MatchingScore(pBlks + 5, true);
			result += MatchingScore(pBlks + 8, false);
		}
		return result;
	}

	PSCORE MatchingPattern2233(const FieldMathingScore* pBlks, int pattern)
	{
    PSCORE result = 0;
		switch (pattern) {
		case 0: // 2233
			result += MatchingScore(pBlks + 0, false);
			result += MatchingScore(pBlks + 2, false);
			result += MatchingScore(pBlks + 4, true);
			result += MatchingScore(pBlks + 7, true);
			break;
		case 1: // 2332
			result += MatchingScore(pBlks + 0, false);
			result += MatchingScore(pBlks + 2, true);
			result += MatchingScore(pBlks + 5, true);
			result += MatchingScore(pBlks + 8, false);
			break;
		case 2: // 3322
			result += MatchingScore(pBlks + 0, true);
			result += MatchingScore(pBlks + 3, true);
			result += MatchingScore(pBlks + 6, false);
			result += MatchingScore(pBlks + 8, false);
			break;
		case 3: // 3223
			result += MatchingScore(pBlks + 0, true);
			result += MatchingScore(pBlks + 3, false);
			result += MatchingScore(pBlks + 5, false);
			result += MatchingScore(pBlks + 7, true);
			break;
		}
		return result;
	}

	void MatchingPattern(const FieldMathingScore* pBlks, PSCORE* score)
	{
		for (int i = 0; i < 5; ++i) {
			int offset = tbl2323[i][0];
			score[i] = MatchingPattern2323(&pBlks[offset + 2], tbl2323[i][1]);
			// 前が空いているパターンは前に３フィールド連続があるはずなのでそれもチェック
			if (offset == 1) {
				score[i] = (score[i] + MatchingScore(pBlks, true)) * 0.8f;
			}
		}
		for (int i = 0; i < 10; ++i) {
			int offset = tbl2233[i][0];
			score[i + 5] = MatchingPattern2233(&pBlks[offset + 2], tbl2233[i][1]);
			// 前が空いているパターンは前に３フィールド連続があるはずなのでそれもチェック
			if (offset == 1) {
				score[i + 5] = (score[i + 5] + MatchingScore(pBlks, true)) * 0.8f;
			}
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
				MatchingPattern(blks, &blockpms[15 * blkidx]);
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
    // 見やすくするために値を小さくする
    for (int i = 0; i < 15; ++i) {
      pms[i] /= fmsPitch;
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
    fmframe->score = curscore;

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

    char buf[100]; sprintf(buf, "KFM: %d (%.1f)", fmframe->pattern, fmframe->score);
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

    DrawInfo(out, fmframe, env);

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

class KTCMerge : public GenericVideoFilter
{
  PClip fmclip;
  PClip clip24;

  const KFMParam* prm;

  VideoInfo tmpvi;

  std::unique_ptr<KTCMergeCoreBase> core;

  struct Frame24Info {
    int index24;     // サイクル内での24pにおけるフレーム番号
    int start60;     // 24pフレームの開始フィールド番号
    bool is3;        // 24pフレームのフィールド数
  };

  KTCMergeCoreBase* CreateCore(IScriptEnvironment* env)
  {
    if (prm->pixelShift == 0) {
      return new KTCMergeCore<uint8_t>(prm);
    }
    else {
      return new KTCMergeCore<uint16_t>(prm);
    }
  }

  Frame24Info GetFrame24Info(int pattern, int frameIndex60)
  {
    if (pattern < 5) {
      int offsets_[] = { -3, 0, 2, 5, 7, 10, 12, 15, 17, 20 };
      int *offsets = &offsets_[1];
      int start24 = tbl2323[pattern][1];
      int offset = frameIndex60 - tbl2323[pattern][0];
      int idx = -1;
      while (offsets[start24 + idx + 1] <= offsets[start24] + offset) idx++;
      int start60 = tbl2323[pattern][0] + offsets[start24 + idx] - offsets[start24];
      int numFields = offsets[start24 + idx + 1] - offsets[start24 + idx];
      Frame24Info ret = { idx, start60, numFields == 3 };
      return ret;
    }
    else {
      int offsets_[] = { -3, 0, 2, 4, 7, 10, 12, 14, 17, 20 };
      int *offsets = &offsets_[1];
      int start24 = tbl2233[pattern - 5][1];
      int offset = frameIndex60 - tbl2233[pattern - 5][0];
      int idx = -1;
      while (offsets[start24 + idx + 1] <= offsets[start24] + offset) idx++;
      int start60 = tbl2233[pattern - 5][0] + offsets[start24 + idx] - offsets[start24];
      int numFields = offsets[start24 + idx + 1] - offsets[start24 + idx];
      Frame24Info ret = { idx, start60, numFields == 3 };
      return ret;
    }
  }

  void GetMatchingScoreFrame(PClip fmclip, int n60, PSCORE *out, IScriptEnvironment* env)
  {
    n60 = clamp(n60, 0, vi.num_frames - 1);

    int cycleIndex = n60 / 10;
    int frameIndex60 = n60 % 10;
    PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
    const KFMFrame* fmframe = (KFMFrame*)fm->GetReadPtr();
    Frame24Info frame24info = GetFrame24Info(fmframe->pattern, frameIndex60);

    // ブロックごとのマッチング度合いを取得
    int fmsPitch = prm->numBlkX * prm->numBlkY;
    const FieldMathingScore* pfms = &fmframe->fms[(frame24info.start60 + 2) * fmsPitch];
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int blkidx = bx + by * prm->numBlkX;
        if (frame24info.is3) {
          FieldMathingScore blks[2];
          blks[0] = pfms[blkidx];
          blks[1] = pfms[blkidx + fmsPitch];
          out[blkidx] = MatchingScore(blks, frame24info.is3);
        }
        else {
          out[blkidx] = MatchingScore(&pfms[blkidx], frame24info.is3);
        }
      }
    }
  }

public:
  KTCMerge(PClip clip60, PClip clip24, PClip fmclip, IScriptEnvironment* env)
    : GenericVideoFilter(clip60)
    , fmclip(fmclip)
    , clip24(clip24)
    , prm(KFMParam::GetParam(fmclip->GetVideoInfo(), env))
    , tmpvi()
  {

    core = std::unique_ptr<KTCMergeCoreBase>(CreateCore(env));

    int numBlks = prm->numBlkX * prm->numBlkY;

    // メモリを確保するデバイスとかマルチスレッドとかメモリ再利用とか考えたくないので、
    // ワークメモリの確保も全部都度NewVideoFrameで行う
    int tmp_frame_bytes = sizeof(PSCORE) * numBlks * 3 + sizeof(bool) * numBlks + core->WorkSize();
    tmpvi.pixel_type = VideoInfo::CS_BGR32;
    tmpvi.width = 2048;
    tmpvi.height = nblocks(tmp_frame_bytes, vi.width * 4);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    int numBlks = prm->numBlkX * prm->numBlkY;

    int cycleIndex = n / 10;
    int frameIndex60 = n % 10;
    PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
    const KFMFrame* fmframe = (KFMFrame*)fm->GetReadPtr();
    Frame24Info frame24info = GetFrame24Info(fmframe->pattern, frameIndex60);

    // メモリ確保
    PVideoFrame outframe = env->NewVideoFrame(vi);
    PVideoFrame tmpframe = env->NewVideoFrame(tmpvi);

    PSCORE *pscore = (PSCORE*)tmpframe->GetWritePtr();
    PSCORE *score = &pscore[numBlks];
    PSCORE *nscore = &score[numBlks];
    bool* match = (bool*)&nscore[numBlks];
    uint8_t* work = (uint8_t*)&match[numBlks];

    // 24pでの前後を含む3フレームのブロックごとのマッチング度合いを取得
    GetMatchingScoreFrame(fmclip, cycleIndex * 10 + frame24info.start60 - 1, pscore, env);
    GetMatchingScoreFrame(fmclip, n, score, env);
    GetMatchingScoreFrame(fmclip, cycleIndex * 10 + frame24info.start60 + (frame24info.is3 ? 3 : 2), nscore, env);

    // 3x3x3の最大値フィルタ適用
    // 時間軸方向
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int blkidx = bx + by * prm->numBlkX;
        pscore[blkidx] = max(pscore[blkidx], max(score[blkidx], nscore[blkidx]));
      }
    }
    // x方向
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int xstart = std::max(0, bx - 1);
        int xend = std::min(prm->numBlkX - 1, bx + 1);
        PSCORE s = 0;
        for (int x = xstart; x < xend; ++x) {
          s = std::max(s, pscore[x + by * prm->numBlkX]);
        }
        nscore[bx + by * prm->numBlkX] = s;
      }
    }
    // y方向
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int ystart = std::max(0, by - 1);
        int yend = std::min(prm->numBlkY - 1, by + 1);
        PSCORE s = 0;
        for (int y = ystart; y < yend; ++y) {
          s = std::max(s, nscore[bx + y * prm->numBlkX]);
        }
        score[bx + by * prm->numBlkX] = s;
      }
    }

    PSCORE threshold = 1000.0f; // TODO:

    int numUnmatchBlks = 0;
    for (int by = 0; by < prm->numBlkY; ++by) {
      for (int bx = 0; bx < prm->numBlkX; ++bx) {
        int blkidx = bx + by * prm->numBlkX;
        match[blkidx] = (score[blkidx] <= threshold);
        if (score[blkidx] > threshold) {
          ++numUnmatchBlks;
        }
      }
    }

    if (numUnmatchBlks >= 0.75 * numBlks) {
      return child->GetFrame(n, env);
    }

    PVideoFrame src24 = clip24->GetFrame(cycleIndex * 4 + frame24info.index24, env);
    PVideoFrame src60 = child->GetFrame(n, env);

    core->Merge(match, src24, src60, outframe, work);

    return outframe;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KTCMerge(
      args[0].AsClip(),       // clip60
      args[1].AsClip(),       // clip24
      args[2].AsClip(),       // fmclip
      env
    );
  }
};

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KFM", "c", KFM::Create, 0);
  env->AddFunction("KIVTC", "cc", KIVTC::Create, 0);
  env->AddFunction("KTCMerge", "ccc", KTCMerge::Create, 0);
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
  init_console();

  AddFuncFM(env);

  return "K Field Matching Plugin";
}
