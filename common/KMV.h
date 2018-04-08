#pragma once

#include <vector>
#include <avisynth.h>

struct VECTOR
{
  int x;
  int y;
  int sad;
};

struct LevelInfo {
  int nBlkX; // number of blocks along X
  int nBlkY; // number of blocks along Y
};

struct KMVParam
{
  enum
  {
    VERSION = 5,
    MAGIC_KEY = 0x4A6C2DE4,
    SUPER_FRAME = 1,
    MV_FRAME = 2,
  };

  /*! \brief Unique identifier, not very useful */
  int nMagicKey; // placed to head in v.1.2.6
  int nVersion; // MVAnalysisData and outfile format version - added in v1.2.6
  int nDataType;

  // Super Frame parameter //

  /*! \brief Width of the frame */
  int nWidth;

  /*! \brief Height of the frame */
  int nHeight;

  // スーパーフレームの縦横と実際の縦横が異なる場合のみ0以外の値が入る
  int nActualWidth;
  int nActualHeight;

  int yRatioUV; // ratio of luma plane height to chroma plane height
  int xRatioUV; // ratio of luma plane height to chroma plane width (fixed to 2 for YV12 and YUY2) PF used!

  int nHPad; // Horizontal padding - v1.8.1
  int nVPad; // Vertical padding - v1.8.1

             /*! \brief pixel refinement of the motion estimation */
  int nPel;

  bool chroma;

  /*! \brief number of level for the hierarchal search */
  int nLevels;
  int nDropLevels;

  int nPixelSize; // PF
  int nBitsPerPixel;
  int nPixelShift;

  int pixelType; // color format

                 // Analyze Frame Parameter //

                 /*! \brief difference between the index of the reference and the index of the current frame */
                 // If nDeltaFrame <= 0, the reference frame is the absolute value of nDeltaFrame.
                 // Only a few functions accept negative nDeltaFrames.
  int nDeltaFrame;

  /*! \brief direction of the search ( forward / backward ) */
  bool isBackward;

  /*! \brief size of a block, in pixel */
  int nBlkSizeX; // horizontal block size
  int nBlkSizeY; // vertical block size - v1.7

  int nOverlapX; // overlap block size - v1.1
  int nOverlapY; // vertical overlap - v1.7

  int nAnalyzeLevels;
  LevelInfo levelInfo[16];

  int chromaSADScale; // P.F. chroma SAD ratio, 0:stay(YV12) 1:div2 2:div4(e.g.YV24)


  KMVParam(int data_type)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , nDataType(data_type)
    , levelInfo()
  { }

  static const KMVParam* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const KMVParam* param = (const KMVParam*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const KMVParam* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};
