#pragma once

#include "Frame.h"

enum {
  OVERLAP = 8,
  VPAD = 4,

  MOVE = 1,
  SHIMA = 2,
  LSHIMA = 4,

  NUM_PATTERNS = 21,
};

struct FMCount {
  int move, shima, lshima;
};

struct PulldownPatternField {
	bool split; // 次のフィールドとは別フレーム
	bool merge; // 3フィールドの最初のフィールド
  bool shift; // 後の24pフレームを参照するフィールド
};

struct PulldownPattern {
  PulldownPatternField fields[10 * 4];
  int cycle;

	PulldownPattern(int nf0, int nf1, int nf2, int nf3); // 24p
	PulldownPattern(); // 30p

  // パターンは10フィールド+前後2フィールドずつの合わせて
  // 14フィールド分をみる想定。14フィールドの前頭へのポインタを返す
  const PulldownPatternField* GetPattern(int n) const {
    return &fields[10 + n - 2];
  }
  int GetCycleLength() const {
    return cycle;
  }
};

struct Frame24Info {
  int cycleIndex;
  int frameIndex; // サイクル内のフレーム番号
  int fieldStartIndex; // ソースフィールド開始番号
  int numFields; // ソースフィールド数
  int fieldShift; // 2224パターンを2323変換する場合のずらしが必要なフレーム
};

struct FMData {
  // 縞と動きの和
  float mft[14];
  float mftr[14];
  float mftcost[14];
};

struct FMMatch {
  float shima[NUM_PATTERNS];
  float costs[NUM_PATTERNS];
  float reliability[NUM_PATTERNS];
};

struct KFMResult {
  int pattern;
  int is60p;
  float score;
  float cost;
  float reliability;

  KFMResult() { }

  KFMResult(int pattern, float score, float cost, float reliability)
    : pattern(pattern)
    , is60p()
    , score(score)
    , cost(cost)
    , reliability(reliability)
  { }

  KFMResult(FMMatch& match, int pattern)
    : pattern(pattern)
    , is60p()
    , score(match.shima[pattern])
    , cost(match.costs[pattern])
    , reliability(match.reliability[pattern])
  { }
};

class PulldownPatterns
{
public:
private:
  PulldownPattern p2323, p2233, p2224, p30;
  int patternOffsets[5];
  const PulldownPatternField* allpatterns[NUM_PATTERNS];
public:
  PulldownPatterns();

  const PulldownPatternField* GetPattern(int patternIndex) const {
    return allpatterns[patternIndex];
  }

  const char* PatternToString(int patternIndex, int& index) const;

  // パターンと24fpsのフレーム番号からフレーム情報を取得
  Frame24Info GetFrame24(int patternIndex, int n24) const;

  // パターンと60fpsのフレーム番号からフレーム情報を取得
  // frameIndex < 0 or frameIndex >= 4の場合、
  // fieldStartIndexとnumFieldsは正しくない可能性があるので注意
  Frame24Info GetFrame60(int patternIndex, int n60) const;

  FMMatch Matching(const FMData& data, int width, int height, float costth, float adj2224, float adj30) const;

  static bool Is30p(int patternIndex) { return patternIndex == NUM_PATTERNS - 1; }
  static bool Is60p(int patternIndex) { return patternIndex == NUM_PATTERNS; }
};

struct CycleAnalyzeInfo {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x6180EDF8,
  };
  int nMagicKey;
  int nVersion;

  int mode;

  CycleAnalyzeInfo(int mode)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , mode(mode)
  { }

  static const CycleAnalyzeInfo* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const CycleAnalyzeInfo* param = (const CycleAnalyzeInfo*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const CycleAnalyzeInfo* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

enum {
  COMBE_FLAG_PAD_W = 4,
  COMBE_FLAG_PAD_H = 2,
};

static Frame WrapSwitchFragFrame(const PVideoFrame& frame) {
  return Frame(frame, COMBE_FLAG_PAD_W, COMBE_FLAG_PAD_H, 1);
}

#define DECOMB_UCF_FLAG_STR "KDecombUCF_Flag"

enum DECOMB_UCF_FLAG {
  DECOMB_UCF_NONE,  // 情報なし
  DECOMB_UCF_PREV,  // 前のフレーム
  DECOMB_UCF_NEXT,  // 次のフレーム
  DECOMB_UCF_FIRST, // 1番目のフィールドでbob
  DECOMB_UCF_SECOND,// 2番目のフィールドでbob
  DECOMB_UCF_NR,    // 汚いフレーム
};

struct DecombUCFInfo {
    enum
    {
        VERSION = 1,
        MAGIC_KEY = 0x7080EDF8,
    };
    int nMagicKey;
    int nVersion;

    int fpsType;

    DecombUCFInfo(int fpsType)
        : nMagicKey(MAGIC_KEY)
        , nVersion(VERSION)
        , fpsType(fpsType)
    { }

    static const DecombUCFInfo* GetParam(const VideoInfo& vi, PNeoEnv env)
    {
        if (vi.sample_type != MAGIC_KEY) {
            env->ThrowError("Invalid source (sample_type signature does not match)");
        }
        const DecombUCFInfo* param = (const DecombUCFInfo*)(void*)vi.num_audio_samples;
        if (param->nMagicKey != MAGIC_KEY) {
            env->ThrowError("Invalid source (magic key does not match)");
        }
        return param;
    }

    static void SetParam(VideoInfo& vi, const DecombUCFInfo* param)
    {
        vi.audio_samples_per_second = 0; // kill audio
        vi.sample_type = MAGIC_KEY;
        vi.num_audio_samples = (size_t)param;
    }
};

int GetDeviceTypes(const PClip& clip);
