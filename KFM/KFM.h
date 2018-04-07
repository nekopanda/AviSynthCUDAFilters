#pragma once

enum {
  OVERLAP = 8,
  BLOCK_SIZE = OVERLAP * 2,
  VPAD = 4,

  MOVE = 1,
  SHIMA = 2,
  LSHIMA = 4,
};

struct FMCount {
  int move, shima, lshima;
};

struct PulldownPatternField {
	bool split; // 次のフィールドとは別フレーム
	bool merge; // 3フィールドの最初のフィールド
};

struct PulldownPattern {
  PulldownPatternField fields[10 * 4];

	PulldownPattern(int nf0, int nf1, int nf2, int nf3);
	PulldownPattern();

  const PulldownPatternField* GetPattern(int n) const {
    return &fields[10 + n - 2];
  }
};

struct Frame24Info {
  int cycleIndex;
  int frameIndex; // サイクル内のフレーム番号
  int fieldStartIndex; // ソースフィールド開始番号
  int numFields; // ソースフィールド数
};

struct FMData;

class PulldownPatterns
{
  PulldownPattern p2323, p2233, p30;
  const PulldownPatternField* allpatterns[27];
public:
  PulldownPatterns();

  const PulldownPatternField* GetPattern(int patternIndex) const {
    return allpatterns[patternIndex];
  }

  // パターンと24fpsのフレーム番号からフレーム情報を取得
  Frame24Info GetFrame24(int patternIndex, int n24) const;

  // パターンと60fpsのフレーム番号からフレーム情報を取得
  // frameIndex < 0 or frameIndex >= 4の場合、
  // fieldStartIndexとnumFieldsは正しくない可能性があるので注意
  Frame24Info GetFrame60(int patternIndex, int n60) const;

  std::pair<int, float> Matching(const FMData* data, int width, int height, float costth) const;

	static bool Is30p(int patternIndex) { return patternIndex >= 18; }
};

#define COMBE_FLAG_STR "KRemoveCombe_Flag"

int GetDeviceTypes(const PClip& clip);
