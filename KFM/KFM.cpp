
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

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

class KIVTCProc
{
	typedef uint16_t pixel_t;

	bool tff;
	int width;
	int height;
	int pixelShift;
	int logUVx;
	int logUVy;

	void CreateWeaveFrame2F(PVideoFrame& srct, PVideoFrame& srcb, PVideoFrame& dst)
	{
		const pixel_t* srctY = (const pixel_t*)srct->GetReadPtr(PLANAR_Y);
		const pixel_t* srctU = (const pixel_t*)srct->GetReadPtr(PLANAR_U);
		const pixel_t* srctV = (const pixel_t*)srct->GetReadPtr(PLANAR_V);
		const pixel_t* srcbY = (const pixel_t*)srcb->GetReadPtr(PLANAR_Y);
		const pixel_t* srcbU = (const pixel_t*)srcb->GetReadPtr(PLANAR_U);
		const pixel_t* srcbV = (const pixel_t*)srcb->GetReadPtr(PLANAR_V);
		pixel_t* dstY = (pixel_t*)dst->GetWritePtr(PLANAR_Y);
		pixel_t* dstU = (pixel_t*)dst->GetWritePtr(PLANAR_U);
		pixel_t* dstV = (pixel_t*)dst->GetWritePtr(PLANAR_V);

		int pitchY = srct->GetPitch(PLANAR_Y);
		int pitchUV = srct->GetPitch(PLANAR_U);
		int widthUV = width >> logUVx;
		int heightUV = height >> logUVy;

		// copy top
		Copy<pixel_t>(dstY, pitchY * 2, srctY, pitchY * 2, width, height / 2);
		Copy<pixel_t>(dstU, pitchUV * 2, srctU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV, pitchUV * 2, srctV, pitchUV * 2, widthUV, heightUV / 2);

		// copy bottom
		Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcbY + pitchY, pitchY * 2, width, height / 2);
		Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcbU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcbV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
	}

	// rfIndex:  0:top, 1:bottom
	void CreateWeaveFrame3F(PVideoFrame& src, PVideoFrame& rf, int rfIndex, PVideoFrame& dst)
	{
		const pixel_t* srcY = (const pixel_t*)src->GetReadPtr(PLANAR_Y);
		const pixel_t* srcU = (const pixel_t*)src->GetReadPtr(PLANAR_U);
		const pixel_t* srcV = (const pixel_t*)src->GetReadPtr(PLANAR_V);
		const pixel_t* rfY = (const pixel_t*)rf->GetReadPtr(PLANAR_Y);
		const pixel_t* rfU = (const pixel_t*)rf->GetReadPtr(PLANAR_U);
		const pixel_t* rfV = (const pixel_t*)rf->GetReadPtr(PLANAR_V);
		pixel_t* dstY = (pixel_t*)dst->GetWritePtr(PLANAR_Y);
		pixel_t* dstU = (pixel_t*)dst->GetWritePtr(PLANAR_U);
		pixel_t* dstV = (pixel_t*)dst->GetWritePtr(PLANAR_V);

		int pitchY = src->GetPitch(PLANAR_Y);
		int pitchUV = src->GetPitch(PLANAR_U);
		int widthUV = width >> logUVx;
		int heightUV = height >> logUVy;

		if (rfIndex == 0) {
			// average top
			Average<pixel_t>(dstY, pitchY * 2, srcY, rfY, pitchY * 2, width, height / 2);
			Average<pixel_t>(dstU, pitchUV * 2, srcU, rfU, pitchUV * 2, widthUV, heightUV / 2);
			Average<pixel_t>(dstV, pitchUV * 2, srcV, rfV, pitchUV * 2, widthUV, heightUV / 2);
			// copy bottom
			Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, pitchY * 2, width, height / 2);
			Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
			Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		}
		else {
			// copy top
			Copy<pixel_t>(dstY, pitchY * 2, srcY, pitchY * 2, width, height / 2);
			Copy<pixel_t>(dstU, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
			Copy<pixel_t>(dstV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);
			// average bottom
			Average<pixel_t>(dstY + pitchY, pitchY * 2, srcY + pitchY, rfY + pitchY, pitchY * 2, width, height / 2);
			Average<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU + pitchUV, rfU + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
			Average<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV + pitchUV, rfV + pitchUV, pitchUV * 2, widthUV, heightUV / 2);
		}
	}

	void CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, PVideoFrame& dst, IScriptEnvironment* env)
	{
		// fstartは0or1にする
		if (fstart < 0 || fstart >= 2) {
			n += fstart / 2;
			fstart &= 1;
		}

		if (fstart == 0 && fnum == 2) {
			dst = clip->GetFrame(n, env);
		}
		else {
			PVideoFrame cur = clip->GetFrame(n, env);
			PVideoFrame nxt = clip->GetFrame(n + 1, env);
			if (fstart == 0 && fnum == 3) {
				CreateWeaveFrame3F(cur, nxt, !tff, dst);
			}
			else if (fstart == 1 && fnum == 3) {
				CreateWeaveFrame3F(nxt, cur, tff, dst);
			}
			else if (fstart == 1 && fnum == 2) {
				if (tff) {
					CreateWeaveFrame2F(nxt, cur, dst);
				}
				else {
					CreateWeaveFrame2F(cur, nxt, dst);
				}
			}
		}
	}

	int GetPattern(PClip fmclip, int cycleNumber, IScriptEnvironment* env)
	{
		PVideoFrame fm = fmclip->GetFrame(cycleNumber, env);
		int pattern = *(int*)fm->GetReadPtr();
		return pattern;
	}

public:

	void GetFrame(PClip clip, PClip fmclip, int n, PVideoFrame& dst, IScriptEnvironment* env)
	{
		int cycleNumber = n / 4;
		int cycleIndex = n % 4;
		int pattern = GetPattern(fmclip, cycleNumber, env);

		int fstart;
		int fnum;

		if (pattern < 5) {
			int offsets[] = { 0, 2, 5, 7, 10, 12, 15, 17, 20 };
			int idx24 = tbl2323[pattern][1];
			fstart = cycleNumber * 10 + tbl2323[pattern][0] +
				(offsets[cycleIndex + idx24] - offsets[idx24]);
			fnum = offsets[cycleIndex + idx24 + 1] - offsets[cycleIndex + idx24];
		}
		else {
			int offsets[] = { 0, 2, 4, 7, 10, 12, 14, 17, 20 };
			int idx24 = tbl2233[pattern - 5][1];
			fstart = cycleNumber * 10 + tbl2233[pattern - 5][0] +
				(offsets[cycleIndex + idx24] - offsets[idx24]);
			fnum = offsets[cycleIndex + idx24 + 1] - offsets[cycleIndex + idx24];
		}

		CreateWeaveFrame(clip, 0, fstart, fnum, dst, env);
	}

	int Get24FrameNumber(PClip clip, int n, IScriptEnvironment* env)
	{
		int cycleNumber = n / 10;
		int cycleIndex = n % 10;
		int pattern = GetPattern(clip, cycleNumber, env);

		if (pattern < 5) {
			int offsets[] = { 0, 2, 5, 7, 10, 12, 15, 17, 20 };
			int start24 = tbl2323[pattern][1];
			int offset = cycleIndex - tbl2323[pattern][0];
			int idx = 0;
			while (offsets[start24 + idx] < offsets[start24] + offset) idx++;
			return cycleNumber * 4 + idx;
		}
		else {
			int offsets[] = { 0, 2, 4, 7, 10, 12, 14, 17, 20 };
			int start24 = tbl2233[pattern - 5][1];
			int offset = cycleIndex - tbl2233[pattern - 5][0];
			int idx = 0;
			while (offsets[start24 + idx] < offsets[start24] + offset) idx++;
			return cycleNumber * 4 + idx;
		}
	}
};

struct FieldMathingScore {
	float n1; // 1フィールド先
	float n2; // 2フィールド先
};

class KFMProc
{
	typedef uint16_t pixel_t;

	bool tff;
	int blockSize;
	int numBlkX;
	int numBlkY;

	int width;
	int height;
	int pixelShift;
	int logUVx;
	int logUVy;

public:
	KFMProc() { }

	void FieldToFrame(PVideoFrame src, PVideoFrame dst, bool isTop)
	{
		const pixel_t* srcY = (const pixel_t*)src->GetReadPtr(PLANAR_Y);
		const pixel_t* srcU = (const pixel_t*)src->GetReadPtr(PLANAR_U);
		const pixel_t* srcV = (const pixel_t*)src->GetReadPtr(PLANAR_V);
		pixel_t* dstY = (pixel_t*)src->GetReadPtr(PLANAR_Y);
		pixel_t* dstU = (pixel_t*)src->GetReadPtr(PLANAR_U);
		pixel_t* dstV = (pixel_t*)src->GetReadPtr(PLANAR_V);

		int pitchY = src->GetPitch(PLANAR_Y);
		int pitchUV = src->GetPitch(PLANAR_U);
		int widthUV = width >> logUVx;
		int heightUV = height >> logUVy;

		if (isTop == false) {
			srcY += pitchY;
			srcU += pitchUV;
			srcV += pitchUV;
		}

		// top field
		Copy<pixel_t>(dstY, pitchY * 2, srcY, pitchY * 2, width, height / 2);
		Copy<pixel_t>(dstU, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);

		// bottom field
		Copy<pixel_t>(dstY + pitchY, pitchY * 2, srcY, pitchY * 2, width, height / 2);
		Copy<pixel_t>(dstU + pitchUV, pitchUV * 2, srcU, pitchUV * 2, widthUV, heightUV / 2);
		Copy<pixel_t>(dstV + pitchUV, pitchUV * 2, srcV, pitchUV * 2, widthUV, heightUV / 2);
	}

	void CompareFieldN1(PVideoFrame base, PVideoFrame ref, bool isTop, FieldMathingScore* fms)
	{
		const pixel_t* baseY = (const pixel_t*)base->GetReadPtr(PLANAR_Y);
		const pixel_t* baseU = (const pixel_t*)base->GetReadPtr(PLANAR_U);
		const pixel_t* baseV = (const pixel_t*)base->GetReadPtr(PLANAR_V);
		const pixel_t* refY = (const pixel_t*)ref->GetReadPtr(PLANAR_Y);
		const pixel_t* refU = (const pixel_t*)ref->GetReadPtr(PLANAR_U);
		const pixel_t* refV = (const pixel_t*)ref->GetReadPtr(PLANAR_V);

		int pitchY = base->GetPitch(PLANAR_Y);
		int pitchUV = base->GetPitch(PLANAR_U);
		int widthUV = width >> logUVx;
		int heightUV = height >> logUVy;

		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int yStart = by * blockSize;
				int yEnd = std::min(yStart + blockSize, height);
				int xStart = bx * blockSize;
				int xEnd = std::min(xStart * blockSize, width);

				float sumY = 0;

				for (int y = yStart + (isTop ? 0 : 1); y < yEnd; y += 2) {
					int y1 = (y == 0) ? 1 : (y - 1);
					int y2 = (y == height - 1) ? (height - 2) : (y + 1);

					for (int x = xStart; x < xEnd; ++x) {
						pixel_t b = baseY[x + y * pitchY];
						pixel_t r1 = refY[x + y1 * pitchY];
						pixel_t r2 = refY[x + y2 * pitchY];
						sumY += (r1 - b) * (r2 - b);
					}
				}

				int yStartUV = yStart >> logUVy;
				int yEndUV = yEnd >> logUVy;
				int xStartUV = xStart >> logUVx;
				int xEndUV = xEnd >> logUVx;

				float sumUV = 0;

				for (int y = yStartUV + (isTop ? 0 : 1); y < yEndUV; y += 2) {
					int y1 = (y == 0) ? 1 : (y - 1);
					int y2 = (y == heightUV - 1) ? (heightUV - 2) : (y + 1);

					for (int x = xStartUV; x < xEndUV; ++x) {
						{
							pixel_t b = baseU[x + y * pitchUV];
							pixel_t r1 = refU[x + y1 * pitchUV];
							pixel_t r2 = refU[x + y2 * pitchUV];
							sumUV += (r1 - b) * (r2 - b);
						}
						{
							pixel_t b = baseV[x + y * pitchUV];
							pixel_t r1 = refV[x + y1 * pitchUV];
							pixel_t r2 = refV[x + y2 * pitchUV];
							sumUV += (r1 - b) * (r2 - b);
						}
					}
				}

				// TODO: chroma scale
				float sum = sumY + sumUV * 1.0f;
				
				// 半端分のスケールを合わせる
				sum *= (float)(blockSize * blockSize) / ((xEnd - xStart) * (yEnd - yStart));

				fms[bx + by * numBlkX].n1 = sum;
			}
		}
	}

	void CompareFieldN2(PVideoFrame base, PVideoFrame ref, bool isTop, FieldMathingScore* fms)
	{
		const pixel_t* baseY = (const pixel_t*)base->GetReadPtr(PLANAR_Y);
		const pixel_t* baseU = (const pixel_t*)base->GetReadPtr(PLANAR_U);
		const pixel_t* baseV = (const pixel_t*)base->GetReadPtr(PLANAR_V);
		const pixel_t* refY = (const pixel_t*)ref->GetReadPtr(PLANAR_Y);
		const pixel_t* refU = (const pixel_t*)ref->GetReadPtr(PLANAR_U);
		const pixel_t* refV = (const pixel_t*)ref->GetReadPtr(PLANAR_V);

		int pitchY = base->GetPitch(PLANAR_Y);
		int pitchUV = base->GetPitch(PLANAR_U);
		int widthUV = width >> logUVx;
		int heightUV = height >> logUVy;

		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int yStart = by * blockSize;
				int yEnd = std::min(yStart + blockSize, height);
				int xStart = bx * blockSize;
				int xEnd = std::min(xStart * blockSize, width);

				float sumY = 0;

				for (int y = yStart + (isTop ? 0 : 1); y < yEnd; y += 2) {
					for (int x = xStart; x < xEnd; ++x) {
						pixel_t b = baseY[x + y * pitchY];
						pixel_t r = refY[x + y * pitchY];
						sumY += (r - b) * (r - b);
					}
				}

				int yStartUV = yStart >> logUVy;
				int yEndUV = yEnd >> logUVy;
				int xStartUV = xStart >> logUVx;
				int xEndUV = xEnd >> logUVx;

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

				// TODO: chroma scale
				float sum = sumY + sumUV * 1.0f;

				// 半端分のスケールを合わせる
				sum *= (float)(blockSize * blockSize) / ((xEnd - xStart) * (yEnd - yStart));

				fms[bx + by * numBlkX].n2 = sum;
			}
		}
	}
};

class KFM : public GenericVideoFilter
{
	VideoInfo srcvi;

	bool tff;
	int blockSize;

	int numBlkX;
	int numBlkY;

	KFMProc proc;

	// パターンスコア
	typedef float PSCORE;

	PSCORE MatchingScore(const FieldMathingScore* pBlk, bool is3)
	{
		if (is3) {
			return (pBlk[0].n1 + pBlk[0].n2 + pBlk[1].n1) / 3;
		}
		return pBlk[0].n1;
	}

	PSCORE MatchingPattern2323(const FieldMathingScore* pBlks, int pattern)
	{
		int result = 0;
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
		int result = 0;
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
				score[i] += MatchingScore(pBlks, true);
			}
		}
		for (int i = 0; i < 10; ++i) {
			int offset = tbl2233[i][0];
			score[i + 5] = MatchingPattern2233(&pBlks[offset + 2], tbl2233[i][1]);
			// 前が空いているパターンは前に３フィールド連続があるはずなのでそれもチェック
			if (offset == 1) {
				score[i] += MatchingScore(pBlks, true);
			}
		}
	}

public:

	KFM(PClip child, IScriptEnvironment* env)
		: GenericVideoFilter(child)
	{
		numBlkX = (vi.width + blockSize - 1) / blockSize;
		numBlkY = (vi.height + blockSize - 1) / blockSize;
	}

	void Analyze(int cycleNumber, IScriptEnvironment* env)
	{
		// 必要なフレームを揃える
		PVideoFrame frames[7];
		for (int i = 0; i < 7; ++i) {
			int fn = cycleNumber * 5 + i - 1;
			if (fn < 0) {
				frames[i] = env->NewVideoFrame(srcvi);
				proc.FieldToFrame(child->GetFrame(0, env), frames[i], tff ? true : false);
			}
			else if (fn >= srcvi.num_frames) {
				frames[i] = env->NewVideoFrame(srcvi);
				proc.FieldToFrame(child->GetFrame(srcvi.num_frames - 1, env), frames[i], tff ? false : true);
			}
			else {
				frames[i] = child->GetFrame(fn, env);
			}
		}

		// ブロックごとのマッチング計算
		FieldMathingScore *fms;
		int fmsPitch = numBlkX * numBlkY;
		for (int i = 0; i < 7; ++i) {
			proc.CompareFieldN1(frames[i], frames[i], tff, &fms[(i * 2 + 0) * fmsPitch]);
			if (i < 6) {
				proc.CompareFieldN2(frames[i], frames[i + 1], tff, &fms[(i * 2 + 0) * fmsPitch]);
				proc.CompareFieldN1(frames[i], frames[i + 1], !tff, &fms[(i * 2 + 1) * fmsPitch]);
				proc.CompareFieldN2(frames[i], frames[i + 1], !tff, &fms[(i * 2 + 1) * fmsPitch]);
			}
		}

		// パターンとのマッチングを計算
		PSCORE *blockpms;
		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int blkidx = bx + by * numBlkX;
				FieldMathingScore blks[14];
				for (int i = 0; i < 14; ++i) {
					blks[i] = fms[i * fmsPitch + blkidx];
				}
				MatchingPattern(blks, &blockpms[15 * blkidx]);
			}
		}

		// パターンごとに合計
		PSCORE pms[15];
		for (int by = 0; by < numBlkY; ++by) {
			for (int bx = 0; bx < numBlkX; ++bx) {
				int blkidx = bx + by * numBlkX;
				for (int i = 0; i < 15; ++i) {
					pms[i] += blockpms[15 * blkidx + i];
				}
			}
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

		// TODO: 出力を書き込む

	}
};

class KTCMergeProc
{
	typedef uint16_t pixel_t;

public:
	KTCMergeProc() { }

};


