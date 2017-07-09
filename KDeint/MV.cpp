#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>
#include "avisynth.h"

#undef min
#undef max



/* returns the biggest integer x such as 2^x <= i */
inline static int ilog2(int i)
{
  int result = 0;
  while (i > 1) { i /= 2; result++; }
  return result;
}

inline static int nblock(int n, int block)
{
  return (n + block - 1) / block;
}


enum { N_PER_BLOCK = 3 };

struct VECTOR
{
  int x;
  int y;
  int sad;
};

struct MVDataGroup
{
  int isValid;
  BYTE data[1]; // MVData[]
};

struct MVData
{
  int nCount;
  VECTOR data[1];
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

  int yRatioUV; // ratio of luma plane height to chroma plane height
  int xRatioUV; // ratio of luma plane height to chroma plane width (fixed to 2 for YV12 and YUY2) PF used!

  int nHPad; // Horizontal padding - v1.8.1
  int nVPad; // Vertical padding - v1.8.1

  /*! \brief pixel refinement of the motion estimation */
  int nPel;

  bool chroma;

  /*! \brief number of level for the hierarchal search */
  int nLevels;

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

  std::vector<LevelInfo> levelInfo;

  int chromaSADScale; // P.F. chroma SAD ratio, 0:stay(YV12) 1:div2 2:div4(e.g.YV24)


  KMVParam(int data_type)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , nDataType(data_type)
  { }

  static const KMVParam* GetParam(const VideoInfo& vi, IScriptEnvironment* env)
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

  static void SetParam(VideoInfo& vi, KMVParam* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

int PlaneHeightLuma(int src_height, int level, int yRatioUV, int vpad)
{
  int height = src_height;

  for (int i = 1; i <= level; i++)
  {
    height = vpad >= yRatioUV ? ((height / yRatioUV + 1) / 2) * yRatioUV : ((height / yRatioUV) / 2) * yRatioUV;
  }
  return height;
}

int PlaneWidthLuma(int src_width, int level, int xRatioUV, int hpad)
{
  int width = src_width;

  for (int i = 1; i <= level; i++)
  {
    width = hpad >= xRatioUV ? ((width / xRatioUV + 1) / 2) * xRatioUV : ((width / xRatioUV) / 2) * xRatioUV;
  }
  return width;
}

unsigned int PlaneSuperOffset(bool chroma, int src_height, int level, int pel, int vpad, int plane_pitch, int yRatioUV)
{
  // storing subplanes in superframes may be implemented by various ways
  int height = src_height; // luma or chroma

  unsigned int offset;

  if (level == 0)
  {
    offset = 0;
  }
  else
  {
    offset = pel*pel*plane_pitch*(src_height + vpad * 2);

    for (int i = 1; i<level; i++)
    {
      height = chroma ? PlaneHeightLuma(src_height*yRatioUV, i, yRatioUV, vpad*yRatioUV) / yRatioUV : PlaneHeightLuma(src_height, i, yRatioUV, vpad);

      offset += plane_pitch*(height + vpad * 2);
    }
  }
  return offset;
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

template<typename pixel_t>
void PadCorner(pixel_t *p, pixel_t v, int hPad, int vPad, int refPitch)
{
  for (int i = 0; i < vPad; i++) {
    if (sizeof(pixel_t) == 1) {
      memset(p, v, hPad); // faster than loop
    }
    else {
      std::fill_n(p, hPad, v);
    }
    p += refPitch;
  }
}

template<typename pixel_t>
void PadFrame(pixel_t *refFrame, int refPitch, int hPad, int vPad, int width, int height)
{
  pixel_t *pfoff = refFrame + vPad * refPitch + hPad;

  // Up-Left
  PadCorner<pixel_t>(refFrame, pfoff[0], hPad, vPad, refPitch);
  // Up-Right
  PadCorner<pixel_t>(refFrame + hPad + width, pfoff[width - 1], hPad, vPad, refPitch);
  // Down-Left
  PadCorner<pixel_t>(refFrame + (vPad + height) * refPitch,
    pfoff[(height - 1) * refPitch], hPad, vPad, refPitch);
  // Down-Right
  PadCorner<pixel_t>(refFrame + hPad + width + (vPad + height) * refPitch,
    pfoff[(height - 1) * refPitch + width - 1], hPad, vPad, refPitch);

  // Top and bottom
  for (int i = 0; i < width; i++)
  {
    pixel_t value_t = pfoff[i];
    pixel_t value_b = pfoff[i + (height - 1) * refPitch];
    pixel_t *p_t = refFrame + hPad + i;
    pixel_t *p_b = p_t + (height + vPad) * refPitch;
    for (int j = 0; j < vPad; j++)
    {
      p_t[0] = value_t;
      p_b[0] = value_b;
      p_t += refPitch;
      p_b += refPitch;
    }
  }

  // Left and right
  for (int i = 0; i < height; i++)
  {
    pixel_t value_l = pfoff[i * refPitch];
    pixel_t value_r = pfoff[i * refPitch + width - 1];
    pixel_t *p_l = refFrame + (vPad + i) * refPitch;
    pixel_t *p_r = p_l + width + hPad;
    for (int j = 0; j < hPad; j++)
    {
      p_l[j] = value_l;
      p_r[j] = value_r;
    }
  }
}

#define RB2_jump(y_new, y, pDst, pSrc, nDstPitch, nSrcPitch) \
{ const int dif = y_new - y; \
  pDst += nDstPitch * dif; \
  pSrc += nSrcPitch * dif * 2; \
  y = y_new; \
}

#define RB2_jump_1(y_new, y, pDst, nDstPitch) \
{ const int dif = y_new - y; \
  pDst += nDstPitch * dif; \
  y = y_new; \
}

// BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing - Fizick
// nHeight is dst height which is reduced by 2 source height
template<typename pixel_t>
void RB2BilinearFilteredVertical(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch,
  int nWidth, int nHeight, int y_beg, int y_end, bool isse2)
{
  const int		y_loop_b = std::max(y_beg, 1);
  const int		y_loop_e = std::min(y_end, nHeight - 1);
  int				y = 0;

  if (y_beg < y_loop_b)
  {
    for (int x = 0; x < nWidth; x++)
    {
      pDst[x] = (pSrc[x] + pSrc[x + nSrcPitch] + 1) / 2;
    }
  }

  RB2_jump(y_loop_b, y, pDst, pSrc, nDstPitch, nSrcPitch);

  for (; y < y_loop_e; ++y)
  {
    for (int x = 0; x < nWidth; x++)
    {
      pDst[x] = (pSrc[x - nSrcPitch]
        + pSrc[x] * 3
        + pSrc[x + nSrcPitch] * 3
        + pSrc[x + nSrcPitch * 2] + 4) / 8;
    }

    pDst += nDstPitch;
    pSrc += nSrcPitch * 2;
  }

  RB2_jump(std::max(y_loop_e, 1), y, pDst, pSrc, nDstPitch, nSrcPitch);

  for (; y < y_end; ++y)
  {
    for (int x = 0; x < nWidth; x++)
    {
      pDst[x] = (pSrc[x] + pSrc[x + nSrcPitch] + 1) / 2;
    }
    pDst += nDstPitch;
    pSrc += nSrcPitch * 2;
  }
}

// BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing - Fizick
// nWidth is dst height which is reduced by 2 source width
template<typename pixel_t>
void RB2BilinearFilteredHorizontalInplace(
  pixel_t *pSrc, int nSrcPitch,
  int nWidth, int nHeight, int y_beg, int y_end, bool isse2)
{
  int				y = 0;

  RB2_jump_1(y_beg, y, pSrc, nSrcPitch);

  for (; y < y_end; ++y)
  {
    int x = 0;
    int pSrc0 = (pSrc[x * 2] + pSrc[x * 2 + 1] + 1) / 2;

    for (int x = 1; x < nWidth - 1; x++)
    {
      pSrc[x] = (pSrc[x * 2 - 1] + pSrc[x * 2] * 3 + pSrc[x * 2 + 1] * 3 + pSrc[x * 2 + 2] + 4) / 8;
    }
    pSrc[0] = pSrc0;

    for (int x = std::max(nWidth - 1, 1); x < nWidth; x++)
    {
      pSrc[x] = (pSrc[x * 2] + pSrc[x * 2 + 1] + 1) / 2;
    }

    pSrc += nSrcPitch;
  }
}

// separable BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing - Fizick v.2.5.2
// assume he have enough horizontal dimension for intermediate results (double as final)
template<typename pixel_t>
void RB2BilinearFiltered(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch,
  int nWidth, int nHeight, int y_beg, int y_end, bool isse2)
{
  RB2BilinearFilteredVertical(pDst, pSrc, nDstPitch, nSrcPitch, nWidth * 2, nHeight, y_beg, y_end, isse2); // intermediate half height
  RB2BilinearFilteredHorizontalInplace(pDst, nDstPitch, nWidth, nHeight, y_beg, y_end, isse2); // inpace width reduction
}

// so called Wiener interpolation. (sharp, similar to Lanczos ?)
// invarint simplified, 6 taps. Weights: (1, -5, 20, 20, -5, 1)/32 - added by Fizick
template<typename pixel_t>
void VerticalWiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
{
  const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

  for (int j = 0; j < 2; j++)
  {
    for (int i = 0; i < nWidth; i++)
      pDst[i] = (pSrc[i] + pSrc[i + nSrcPitch] + 1) >> 1;
    pDst += nDstPitch;
    pSrc += nSrcPitch;
  }
  for (int j = 2; j < nHeight - 4; j++)
  {
    for (int i = 0; i < nWidth; i++)
    {
      pDst[i] = std::min(max_pixel_value, std::max(0,
        ((pSrc[i - nSrcPitch * 2])
          + (-(pSrc[i - nSrcPitch]) + (pSrc[i] << 2) + (pSrc[i + nSrcPitch] << 2) - (pSrc[i + nSrcPitch * 2])) * 5
          + (pSrc[i + nSrcPitch * 3]) + 16) >> 5));
    }
    pDst += nDstPitch;
    pSrc += nSrcPitch;
  }
  for (int j = nHeight - 4; j < nHeight - 1; j++)
  {
    for (int i = 0; i < nWidth; i++)
    {
      pDst[i] = (pSrc[i] + pSrc[i + nSrcPitch] + 1) >> 1;
    }

    pDst += nDstPitch;
    pSrc += nSrcPitch;
  }
  // last row
  for (int i = 0; i < nWidth; i++)
    pDst[i] = pSrc[i];
}

template<typename pixel_t>
void HorizontalWiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
{
  const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

  for (int j = 0; j < nHeight; j++)
  {
    pDst[0] = (pSrc[0] + pSrc[1] + 1) >> 1;
    pDst[1] = (pSrc[1] + pSrc[2] + 1) >> 1;
    for (int i = 2; i < nWidth - 4; i++)
    {
      pDst[i] = std::min(max_pixel_value, std::max(0, ((pSrc[i - 2]) + (-(pSrc[i - 1]) + (pSrc[i] << 2)
        + (pSrc[i + 1] << 2) - (pSrc[i + 2])) * 5 + (pSrc[i + 3]) + 16) >> 5));
    }
    for (int i = nWidth - 4; i < nWidth - 1; i++)
      pDst[i] = (pSrc[i] + pSrc[i + 1] + 1) >> 1;

    pDst[nWidth - 1] = pSrc[nWidth - 1];
    pDst += nDstPitch;
    pSrc += nSrcPitch;
  }
}

class KMPlaneBase {
public:
  //virtual void SetInterp(int nRfilter, int nSharp) = 0;
  virtual void SetTarget(uint8_t* pSrc, int _nPitch) = 0;
  virtual void Fill(const uint8_t *_pNewPlane, int nNewPitch) = 0;
  virtual void Pad() = 0;
  virtual void Refine() = 0;
  virtual void ReduceTo(KMPlaneBase* dstPlane) = 0;
};

template <typename pixel_t>
class KMPlane : public KMPlaneBase
{
  std::unique_ptr<pixel_t*[]> pPlane;
  int nPel;
  int nWidth;
  int nHeight;
  int nPitch;
  int nHPad;
  int nVPad;
  int nOffsetPadding;
  int nHPadPel;
  int nVPadPel;
  int nExtendedWidth;
  int nExtendedHeight;
  int nBitsPerPixel;
public:

  KMPlane(int nWidth, int nHeight, int nPel, int nHPad, int nVPad, int nBitsPerPixel, bool isse, bool mt_flag)
    : pPlane(new pixel_t*[nPel * nPel])
    , nPel(nPel)
    , nWidth(nWidth)
    , nHeight(nHeight)
    , nPitch(0)
    , nHPad(nHPad)
    , nVPad(nVPad)
    , nOffsetPadding(0)
    , nHPadPel(nHPad * nPel)
    , nVPadPel(nVPad * nPel)
    , nExtendedWidth(nWidth + 2 * nHPad)
    , nExtendedHeight(nHeight + 2 * nVPad)
    , nBitsPerPixel(nBitsPerPixel)
  {
    //
  }

  int GetNPel() const { return nPel; }
  int GetPitch() const { return nPitch; }
  int GetWidth() const { return nWidth; }
  int GetHeight() const { return nHeight; }
  int GetExtendedWidth() const { return nExtendedWidth; }
  int GetExtendedHeight() const { return nExtendedHeight; }
  int GetHPadding() const { return nHPad; }
  int GetVPadding() const { return nVPad; }

  template <int NPELL2>
  const pixel_t *GetAbsolutePointerPel(int nX, int nY) const
  {
    enum { MASK = (1 << NPELL2) - 1 };

    int idx = (nX & MASK) | ((nY & MASK) << NPELL2);

    nX >>= NPELL2;
    nY >>= NPELL2;

    return pPlane[idx] + nX + nY * nPitch;
  }

  template <>
  const pixel_t *GetAbsolutePointerPel <0>(int nX, int nY) const
  {
    return pPlane[0] + nX + nY * nPitch;
  }

  const pixel_t *GetAbsolutePointer(int nX, int nY) const
  {
    if (nPel == 1)
    {
      return GetAbsolutePointerPel <0>(nX, nY);
    }
    else if (nPel == 2)
    {
      return GetAbsolutePointerPel <1>(nX, nY);
    }
    else // nPel == 4
    {
      return GetAbsolutePointerPel <2>(nX, nY);
    }
  }

  const pixel_t *GetAbsolutePelPointer(int nX, int nY) const
  {
    return pPlane[0] + nX + nY * nPitch;
  }

  //void SetInterp(int nRfilter, int nSharp)
  //{
  //  // 今は未対応
  //}

  void SetTarget(uint8_t* _pSrc, int _nPitch)
  {
    pixel_t* pSrc = (pixel_t*)_pSrc;
    nPitch = _nPitch;
    nOffsetPadding = nPitch * nVPad + nHPad;

    for (int i = 0; i < nPel * nPel; i++)
    {
      pPlane[i] = pSrc + i * nPitch * nExtendedHeight;
    }
  }

  void Fill(const uint8_t *_pNewPlane, int nNewPitch)
  {
    const pixel_t* pNewPlane = (const pixel_t*)_pNewPlane;

    Copy(pPlane[0] + nOffsetPadding, nPitch, pNewPlane, nNewPitch, nWidth, nHeight);
  }

  void Pad()
  {
    PadFrame(pPlane[0], nPitch, nHPad, nVPad, nWidth, nHeight);
  }

  void Refine()
  {
    switch (nPel)
    {
    case 2:
      HorizontalWiener(pPlane[1], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, nBitsPerPixel);
      VerticalWiener(pPlane[2], pPlane[0], nPitch, nPitch, nExtendedWidth, nExtendedHeight, nBitsPerPixel);
      HorizontalWiener(pPlane[3], pPlane[2], nPitch, nPitch, nExtendedWidth, nExtendedHeight, nBitsPerPixel);
      break;
    //case 4: // 今は未対応
    //  break;
    }
  }

  void ReduceTo(KMPlaneBase* dstPlane)
  {
    KMPlane<pixel_t>&		red = *static_cast<KMPlane<pixel_t>*>(dstPlane);
    RB2BilinearFiltered(
      red.pPlane[0] + red.nOffsetPadding, pPlane[0] + nOffsetPadding,
      red.nPitch, nPitch,
      red.nWidth, red.nHeight, 0, red.nHeight,
      true
    );
  }
};

class KMFrame
{
  const KMVParam* param;

  std::unique_ptr<KMPlaneBase> pYPlane;
  std::unique_ptr<KMPlaneBase> pUPlane;
  std::unique_ptr<KMPlaneBase> pVPlane;

  template <typename pixel_t>
  void CreatePlanes(int nWidth, int nHeight, int nPel, int nHPad, int nVPad, bool mt_flag)
  {
    pYPlane = std::unique_ptr<KMPlaneBase>(new KMPlane<uint8_t>(
      nWidth, nHeight, nPel, nHPad, nVPad, param->nBitsPerPixel, true, mt_flag));
    if (param->chroma) {
      pUPlane = std::unique_ptr<KMPlaneBase>(new KMPlane<uint8_t>(
        nWidth / param->xRatioUV, nHeight / param->yRatioUV, nPel,
        nHPad / param->xRatioUV, nVPad / param->yRatioUV, param->nBitsPerPixel, true, mt_flag));
      pVPlane = std::unique_ptr<KMPlaneBase>(new KMPlane<uint8_t>(
        nWidth / param->xRatioUV, nHeight / param->yRatioUV, nPel,
        nHPad / param->xRatioUV, nVPad / param->yRatioUV, param->nBitsPerPixel, true, mt_flag));
    }
  }

public:
  KMFrame(int nWidth, int nHeight, int nPel, const KMVParam* param)
    : param(param)
  {
    if (param->nPixelSize == 1) {
      CreatePlanes<uint8_t>(nWidth, nHeight, nPel, param->nHPad, param->nVPad, false);
    }
    else if (param->nPixelSize == 2) {
      CreatePlanes<uint16_t>(nWidth, nHeight, nPel, param->nHPad, param->nVPad, false);
    }
  }

  KMPlaneBase* GetYPlane() { return pYPlane.get(); }
  KMPlaneBase* GetUPlane() { return pUPlane.get(); }
  KMPlaneBase* GetVPlane() { return pVPlane.get(); }

  void SetTarget(uint8_t * pSrcY, int pitchY, uint8_t * pSrcU, int pitchU, uint8_t *pSrcV, int pitchV)
  {
    pYPlane->SetTarget(pSrcY, pitchY);
    if (param->chroma) {
      pUPlane->SetTarget(pSrcU, pitchU);
      pVPlane->SetTarget(pSrcV, pitchV);
    }
  }

  void Fill(const uint8_t * pSrcY, int pitchY, const uint8_t * pSrcU, int pitchU, const uint8_t *pSrcV, int pitchV)
  {
    pYPlane->Fill(pSrcY, pitchY);
    if (param->chroma) {
      pUPlane->Fill(pSrcU, pitchU);
      pVPlane->Fill(pSrcV, pitchV);
    }
  }

  //void	SetInterp(int rfilter, int sharp)
  //{
  //  pYPlane->SetInterp(rfilter, sharp);
  //  if (chroma) {
  //    pUPlane->SetInterp(rfilter, sharp);
  //    pVPlane->SetInterp(rfilter, sharp);
  //  }
  //}

  void	Refine()
  {
    pYPlane->Refine();
    if (param->chroma) {
      pUPlane->Refine();
      pVPlane->Refine();
    }
  }

  void	Pad()
  {
    pYPlane->Pad();
    if (param->chroma) {
      pUPlane->Pad();
      pVPlane->Pad();
    }
  }

  void	ReduceTo(KMFrame *pFrame)
  {
    pYPlane->ReduceTo(pFrame->GetYPlane());
    if (param->chroma) {
      pUPlane->ReduceTo(pFrame->GetUPlane());
      pVPlane->ReduceTo(pFrame->GetVPlane());
    }
  }
};

class KMSuperFrame
{
  const KMVParam* param;
  std::unique_ptr<std::unique_ptr<KMFrame>[]> pFrames;

public:
  // xRatioUV PF 160729
  KMSuperFrame(const KMVParam* param)
    : param(param)
    , pFrames(new std::unique_ptr<KMFrame>[param->nLevels])
  {
    pFrames[0] = std::unique_ptr<KMFrame>(new KMFrame(param->nWidth, param->nHeight, param->nPel, param));
    for (int i = 1; i < param->nLevels; i++)
    {
      int nWidthi = PlaneWidthLuma(param->nWidth, i, param->xRatioUV, param->nHPad);
      int nHeighti = PlaneHeightLuma(param->nHeight, i, param->yRatioUV, param->nVPad);
      pFrames[i] = std::unique_ptr<KMFrame>(new KMFrame(nWidthi, nHeighti, 1, param));
    }
  }

  void SetTarget(uint8_t * pSrcY, int pitchY, uint8_t * pSrcU, int pitchU, uint8_t *pSrcV, int pitchV)
  {
    for (int i = 0; i < param->nLevels; i++)
    {
      unsigned int offY = PlaneSuperOffset(
        false, param->nHeight, i, param->nPel, param->nVPad, pitchY, param->yRatioUV); // no need here xRatioUV and pixelsize
      unsigned int offU = PlaneSuperOffset(
        true, param->nHeight / param->yRatioUV, i, param->nPel, param->nVPad / param->yRatioUV, pitchU, param->yRatioUV);
      unsigned int offV = PlaneSuperOffset(
        true, param->nHeight / param->yRatioUV, i, param->nPel, param->nVPad / param->yRatioUV, pitchV, param->yRatioUV);
      pFrames[i]->SetTarget(
        pSrcY + offY * param->nPixelSize, pitchY, pSrcU + offU * param->nPixelSize, pitchU, pSrcV + offV * param->nPixelSize, pitchV);
    }
  }

  void Construct(const uint8_t * pSrcY, int pitchY, const uint8_t * pSrcU, int pitchU, const uint8_t *pSrcV, int pitchV)
  {
    pFrames[0]->Fill(pSrcY, pitchY, pSrcU, pitchU, pSrcV, pitchV);
    pFrames[0]->Pad();
    pFrames[0]->Refine();

    for (int i = 0; i < param->nLevels - 1; i++)
    {
      pFrames[i]->ReduceTo(pFrames[i + 1].get());
      pFrames[i + 1]->Pad();
    }
  }

  KMFrame *GetFrame(int nLevel)
  {
    if ((nLevel < 0) || (nLevel >= param->nLevels)) {
      return nullptr;
    }
    return pFrames[nLevel].get();
  }

  //void	SetInterp(int rfilter, int sharp)
  //{
  //  // TODO: これなんでゼロだけ？
  //  pFrames[0]->SetInterp(rfilter, sharp);
  //}
};

class KMSuper : public GenericVideoFilter
{
  // 引数パラメータ
  int nHPad;
  int nVPad;
  int nPel;
  int nLevels;
  bool chroma;
  int nSharp;
  int nRfilter;

  // その他
  int nSuperWidth;
  int nSuperHeight;

  KMVParam params;

  std::unique_ptr<KMSuperFrame> pSrcGOF;

public:
  KMSuper(PClip child, int debug, IScriptEnvironment* env)
    : GenericVideoFilter(child)
    , params(KMVParam::SUPER_FRAME)
  {
    // 今の所対応しているのコレだけ
    nHPad = 8;
    nVPad = 8;
    nPel = 2;
    nLevels = 0;
    chroma = true;
    nSharp = 2;
    nRfilter = 2;
    
    params.nWidth = vi.width;
    params.nHeight = vi.height;
    params.yRatioUV = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
    params.xRatioUV = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);

    params.nPixelSize = vi.ComponentSize();
    params.nBitsPerPixel = vi.BitsPerComponent();
    params.nPixelShift = (params.nPixelSize == 1) ? 0 : 1;

    int nLevelsMax = 0;
    while (PlaneHeightLuma(vi.height, nLevelsMax, params.yRatioUV, nVPad) >= params.yRatioUV * 2 &&
      PlaneWidthLuma(vi.width, nLevelsMax, params.xRatioUV, nHPad) >= params.xRatioUV * 2) // at last two pixels width and height of chroma
    {
      nLevelsMax++;
    }
    if (nLevels <= 0 || nLevels > nLevelsMax) {
      nLevels = nLevelsMax;
    }

    nSuperWidth = params.nWidth + 2 * nHPad;
    nSuperHeight = PlaneSuperOffset(false, params.nHeight, nLevels, nPel, nVPad, nSuperWidth, params.yRatioUV) / nSuperWidth;
    if (params.yRatioUV == 2 && nSuperHeight & 1) {
      nSuperHeight++; // even
    }
    vi.width = nSuperWidth;
    vi.height = nSuperHeight;

    params.nHPad = nHPad;
    params.nVPad = nVPad;
    params.nPel = nPel;
    params.chroma = chroma;
    params.nLevels = nLevels;
    params.pixelType = vi.pixel_type;

    KMVParam::SetParam(vi, &params);

    pSrcGOF = std::unique_ptr<KMSuperFrame>(new KMSuperFrame(&params));

    //pSrcGOF->SetInterp(nRfilter, nSharp);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame	dst = env->NewVideoFrame(vi);

    const BYTE* pSrcY = src->GetReadPtr(PLANAR_Y);
    const BYTE* pSrcU = src->GetReadPtr(PLANAR_U);
    const BYTE* pSrcV = src->GetReadPtr(PLANAR_V);
    int nSrcPitchY = src->GetPitch(PLANAR_Y) >> params.nPixelShift;
    int nSrcPitchUV = src->GetPitch(PLANAR_U) >> params.nPixelShift;

    BYTE* pDstY = dst->GetWritePtr(PLANAR_Y);
    BYTE* pDstU = dst->GetWritePtr(PLANAR_U);
    BYTE* pDstV = dst->GetWritePtr(PLANAR_V);
    int nDstPitchY = dst->GetPitch(PLANAR_Y) >> params.nPixelShift;
    int nDstPitchUV = dst->GetPitch(PLANAR_U) >> params.nPixelShift;

    pSrcGOF->SetTarget(pDstY, nDstPitchY, pDstU, nDstPitchUV, pDstV, nDstPitchUV);
    pSrcGOF->Construct(pSrcY, nSrcPitchY, pSrcU, nSrcPitchUV, pSrcV, nSrcPitchUV);

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KMSuper(args[0].AsClip(), args[1].AsInt(0), env);
  }
};


/*! \brief Search type : defines the algorithm used for minimizing the SAD */
enum SearchType
{
  ONETIME = 1,
  NSTEP = 2,
  LOGARITHMIC = 4,
  EXHAUSTIVE = 8,
  HEX2SEARCH = 16,   // v.2
  UMHSEARCH = 32,   // v.2
  HSEARCH = 64,   // v.2.5.11
  VSEARCH = 128   // v.2.5.11
};

static unsigned int SADABS(int x) { return (x < 0) ? -x : x; }

template<int nBlkWidth, int nBlkHeight, typename pixel_t>
static unsigned int Sad_C(const pixel_t *pSrc, int nSrcPitch, const pixel_t *pRef, int nRefPitch)
{
  unsigned int sum = 0; // int is probably enough for 32x32
  for (int y = 0; y < nBlkHeight; y++)
  {
    for (int x = 0; x < nBlkWidth; x++)
      sum += SADABS(pSrc[x] - pRef[x]);
    pSrc += nSrcPitch;
    pRef += nRefPitch;
  }
  return sum;
}

template<typename pixel_t>
unsigned int(*get_sad_function(int nBlkWidth, int nBlkHeight, IScriptEnvironment* env))(const pixel_t *pSrc, int nSrcPitch, const pixel_t *pRef, int nRefPitch)
{
  if (nBlkWidth == 8 && nBlkHeight == 8) {
    return Sad_C<8, 8, pixel_t>;
  }
  else if (nBlkWidth == 16 && nBlkHeight == 16) {
    return Sad_C<16, 16, pixel_t>;
  }
  else if (nBlkWidth == 32 && nBlkHeight == 32) {
    return Sad_C<32, 32, pixel_t>;
  }
  else {
    env->ThrowError("Not supported blocksize (%d,%d)", nBlkWidth, nBlkHeight);
  }
  return nullptr;
}

class PlaneOfBlocksBase {
public:
  virtual void EstimateGlobalMVDoubled(VECTOR* globalMV) = 0;
  virtual void InterpolatePrediction(const PlaneOfBlocksBase* pob) = 0;
  virtual void SearchMVs(KMFrame *pSrcFrame, KMFrame *pRefFrame, const VECTOR *_globalMV, MVData *out) = 0;
  virtual int GetArraySize() = 0;
  virtual int WriteDefault(MVData *out) = 0;
};

template <typename pixel_t>
class PlaneOfBlocks : public PlaneOfBlocksBase
{
  /* fields set at initialization */

  const int      nBlkX;            /* width in number of blocks */
  const int      nBlkY;            /* height in number of blocks */
  const int      nBlkSizeX;        /* size of a block */
  const int      nBlkSizeY;        /* size of a block */
  const int      nBlkCount;        /* number of blocks in the plane */
  const int      nPel;             /* pel refinement accuracy */
  const int      nLogPel;          /* logarithm of the pel refinement accuracy */
  const int      nScale;           /* scaling factor of the plane */
  const int      nLogScale;        /* logarithm of the scaling factor */
  const bool     smallestPlane;
  const bool     chroma;            /* do we do chroma me */
  const int      nOverlapX;        // overlap size
  const int      nOverlapY;        // overlap size
  const int      xRatioUV;        // PF
  const int      nLogxRatioUV;     // log of xRatioUV (0 for 1 and 1 for 2)
  const int      yRatioUV;
  const int      nLogyRatioUV;     // log of yRatioUV (0 for 1 and 1 for 2)
  const int      nPixelSize; // PF
  const int      nPixelShift; // log of pixelsize (0,1,2) for shift instead of mul or div
  const int      nBitsPerPixel;
  const bool     _mt_flag;         // Allows multithreading
  const int      chromaSADscale;   // PF experimental 2.7.18.22 allow e.g. YV24 chroma to have the same magnitude as for YV12

  const SearchType searchType;
  const int nSearchParam;
  const int PelSearch;
  const int lsad;
  const int penaltyNew;
  const int plevel;
  const bool global;
  const int penaltyZero;
  const int pglobal;
  const int badSAD;
  const int badrange;
  const bool meander;
  const bool tryMany;

  const int verybigSAD;
  int nLambdaLevel;

  unsigned int (* const SAD)(const pixel_t *pSrc, int nSrcPitch, const pixel_t *pRef, int nRefPitch);
  unsigned int (* const SADCHROMA)(const pixel_t *pSrc, int nSrcPitch, const pixel_t *pRef, int nRefPitch);

  std::vector <VECTOR>              /* motion vectors of the blocks */
    vectors;           /* before the search, contains the hierachal predictor */
                       /* after the search, contains the best motion vector */

  KMPlane<pixel_t> *pSrcYPlane, *pSrcUPlane, *pSrcVPlane;
  KMPlane<pixel_t> *pRefYPlane, *pRefUPlane, *pRefVPlane;

  int nSrcPitch[3];
  int nRefPitch[3];

  int x[3];                   /* absolute x coordinate of the origin of the block in the reference frame */
  int y[3];                   /* absolute y coordinate of the origin of the block in the reference frame */
  int blkx;                   /* x coordinate in blocks */
  int blky;                   /* y coordinate in blocks */
  int blkIdx;                 /* index of the block */
  int blkScanDir;             // direction of scan (1 is left to right, -1 is right to left)

  VECTOR globalMVPredictor;   // predictor of global motion vector

                              // Current block
  const pixel_t* pSrc[3];     // the alignment of this array is important for speed for some reason (cacheline?)

  VECTOR bestMV;              /* best vector found so far during the search */
  int nMinCost;               /* minimum cost ( sad + mv cost ) found so far */
  VECTOR predictor;           /* best predictor for the current vector */
  VECTOR predictors[5];   /* set of predictors for the current block */

  int nDxMin;                 /* minimum x coordinate for the vector */
  int nDyMin;                 /* minimum y coordinate for the vector */
  int nDxMax;                 /* maximum x corrdinate for the vector */
  int nDyMax;                 /* maximum y coordinate for the vector */

  int nCurrentLambda;                /* vector cost factor */
  int iter;                   // MOTION_DEBUG only?
  int srcLuma;

  /* computes square distance between two vectors */
  static unsigned int SquareDifferenceNorm(const VECTOR& v1, const int v2x, const int v2y)
  {
    return (v1.x - v2x) * (v1.x - v2x) + (v1.y - v2y) * (v1.y - v2y);
  }

  bool IsVectorOK(int vx, int vy) const
  {
    return (
      (vx >= nDxMin)
      && (vy >= nDyMin)
      && (vx < nDxMax)
      && (vy < nDyMax)
      );
  }

  int MotionDistorsion(int vx, int vy) const
  {
    int dist = SquareDifferenceNorm(predictor, vx, vy);
    if (sizeof(pixel_t) == 1)
      return (nCurrentLambda * dist) >> 8; // 8 bit: faster
    else
      return (nCurrentLambda * dist) >> (16 - nBitsPerPixel) /*8*/; // PF scaling because it appears as a sad addition 
  }

  /* fetch the block in the reference frame, which is pointed by the vector (vx, vy) */
  const pixel_t *	GetRefBlock(int nVx, int nVy)
  {
    return (nPel == 2) ? pRefYPlane->GetAbsolutePointerPel <1>((x[0] << 1) + nVx, (y[0] << 1) + nVy) :
      (nPel == 1) ? pRefYPlane->GetAbsolutePointerPel <0>((x[0]) + nVx, (y[0]) + nVy) :
      pRefYPlane->GetAbsolutePointerPel <2>((x[0] << 2) + nVx, (y[0] << 2) + nVy);
  }

  const pixel_t *	GetRefBlockU(int nVx, int nVy)
  {
    return (nPel == 2) ? pRefUPlane->GetAbsolutePointerPel <1>((x[1] << 1) + (nVx >> nLogxRatioUV), (y[1] << 1) + (nVy >> nLogyRatioUV)) :
      (nPel == 1) ? pRefUPlane->GetAbsolutePointerPel <0>((x[1]) + (nVx >> nLogxRatioUV), (y[1]) + (nVy >> nLogyRatioUV)) :
      pRefUPlane->GetAbsolutePointerPel <2>((x[1] << 2) + (nVx >> nLogxRatioUV), (y[1] << 2) + (nVy >> nLogyRatioUV));
  }

  const pixel_t *	GetRefBlockV(int nVx, int nVy)
  {
    return (nPel == 2) ? pRefVPlane->GetAbsolutePointerPel <1>((x[2] << 1) + (nVx >> nLogxRatioUV), (y[2] << 1) + (nVy >> nLogyRatioUV)) :
      (nPel == 1) ? pRefVPlane->GetAbsolutePointerPel <0>((x[2]) + (nVx >> nLogxRatioUV), (y[2]) + (nVy >> nLogyRatioUV)) :
      pRefVPlane->GetAbsolutePointerPel <2>((x[2] << 2) + (nVx >> nLogxRatioUV), (y[2] << 2) + (nVy >> nLogyRatioUV));
  }

  /* clip a vector to the horizontal boundaries */
  int	ClipMVx(int vx)
  {
    //	return imin(nDxMax - 1, imax(nDxMin, vx));
    if (vx < nDxMin) return nDxMin;
    else if (vx >= nDxMax) return nDxMax - 1;
    else return vx;
  }

  /* clip a vector to the vertical boundaries */
  int	ClipMVy(int vy)
  {
    //	return imin(nDyMax - 1, imax(nDyMin, vy));
    if (vy < nDyMin) return nDyMin;
    else if (vy >= nDyMax) return nDyMax - 1;
    else return vy;
  }

  /* clip a vector to the search boundaries */
  VECTOR ClipMV(VECTOR v)
  {
    VECTOR v2;
    v2.x = ClipMVx(v.x);
    v2.y = ClipMVy(v.y);
    v2.sad = v.sad;
    return v2;
  }

  /* find the median between a, b and c */
  int	Median(int a, int b, int c)
  {
    //	return a + b + c - imax(a, imax(b, c)) - imin(c, imin(a, b));
    if (a < b)
    {
      if (b < c) return b;
      else if (a < c) return c;
      else return a;
    }
    else {
      if (a < c) return a;
      else if (b < c) return c;
      else return b;
    }
  }

  void FetchPredictors()
  {
    VECTOR zero = { 0,0,0 };

    // Left (or right) predictor
    if (false) {
      if ((blkScanDir == 1 && blkx >= 2) || (blkScanDir == -1 && blkx < nBlkX - 2))
      {
        int diff = -1;
        if (blkScanDir == 1) {
          diff = blkx - ((blkx & ~1) - 1);
        }
        predictors[1] = ClipMV(vectors[blkIdx - diff]);
      }
      else
      {
        predictors[1] = ClipMV(zero); // v1.11.1 - values instead of pointer
      }
    }
    else if ((blkScanDir == 1 && blkx > 0) || (blkScanDir == -1 && blkx < nBlkX - 1))
    {
      predictors[1] = ClipMV(vectors[blkIdx - blkScanDir]);
    }
    else
    {
      predictors[1] = ClipMV(zero); // v1.11.1 - values instead of pointer
    }

    // Up predictor
    if (blky > 0)
    {
      predictors[2] = ClipMV(vectors[blkIdx - nBlkX]);
    }
    else
    {
      predictors[2] = ClipMV(zero);
    }

    // bottom-right pridictor (from coarse level)
    if ((blky < nBlkY - 1) && ((blkScanDir == 1 && blkx < nBlkX - 1) || (blkScanDir == -1 && blkx > 0)))
    {
      predictors[3] = ClipMV(vectors[blkIdx + nBlkX + blkScanDir]);
    }
    // Up-right predictor
    else if ((blky > 0) && ((blkScanDir == 1 && blkx < nBlkX - 1) || (blkScanDir == -1 && blkx > 0)))
    {
      predictors[3] = ClipMV(vectors[blkIdx - nBlkX + blkScanDir]);
    }
    else
    {
      predictors[3] = ClipMV(zero);
    }

    // Median predictor
    if (blky > 0) // replaced 1 by 0 - Fizick
    {
      predictors[0].x = Median(predictors[1].x, predictors[2].x, predictors[3].x);
      predictors[0].y = Median(predictors[1].y, predictors[2].y, predictors[3].y);
      //		predictors[0].sad = Median(predictors[1].sad, predictors[2].sad, predictors[3].sad);
      // but it is not true median vector (x and y may be mixed) and not its sad ?!
      // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
      predictors[0].sad = std::max(predictors[1].sad, std::max(predictors[2].sad, predictors[3].sad));
    }
    else
    {
      // but for top line we have only left predictor[1] - v1.6.0
      predictors[0].x = predictors[1].x;
      predictors[0].y = predictors[1].y;
      predictors[0].sad = predictors[1].sad;
    }

    // if there are no other planes, predictor is the median
    if (smallestPlane)
    {
      predictor = predictors[0];
    }

    typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64 >::type safe_sad_t;
    nCurrentLambda = nCurrentLambda*(safe_sad_t)lsad / ((safe_sad_t)lsad + (predictor.sad >> 1))*lsad / ((safe_sad_t)lsad + (predictor.sad >> 1));
    // replaced hard threshold by soft in v1.10.2 by Fizick (a liitle complex expression to avoid overflow)
    //	int a = LSAD/(LSAD + (predictor.sad>>1));
    //	nCurrentLambda = nCurrentLambda*a*a;
  }

  void ExpandingSearch(int r, int s, int mvx, int mvy) // diameter = 2*r + 1, step=s
  { // part of true enhaustive search (thin expanding square) around mvx, mvy
    int i, j;
    //	VECTOR mv = bestMV; // bug: it was pointer assignent, not values, so iterative! - v2.1

    // sides of square without corners
    for (i = -r + s; i < r; i += s) // without corners! - v2.1
    {
      CheckMV<false>(mvx + i, mvy - r);
      CheckMV<false>(mvx + i, mvy + r);
    }

    for (j = -r + s; j < r; j += s)
    {
      CheckMV<false>(mvx - r, mvy + j);
      CheckMV<false>(mvx + r, mvy + j);
    }

    // then corners - they are more far from cenrer
    CheckMV<false>(mvx - r, mvy - r);
    CheckMV<false>(mvx - r, mvy + r);
    CheckMV<false>(mvx + r, mvy - r);
    CheckMV<false>(mvx + r, mvy + r);
  }

  void Hex2Search(int i_me_range)
  {
    /* (x-1)%6 */
    static const int mod6m1[8] = { 5,0,1,2,3,4,5,0 };
    /* radius 2 hexagon. repeated entries are to avoid having to compute mod6 every time. */
    static const int hex2[8][2] = { { -1,-2 },{ -2,0 },{ -1,2 },{ 1,2 },{ 2,0 },{ 1,-2 },{ -1,-2 },{ -2,0 } };

    // adopted from x264
    int dir = -2;
    int bmx = bestMV.x;
    int bmy = bestMV.y;

    if (i_me_range > 1)
    {
      /* hexagon */
      CheckMVdir(bmx - 2, bmy, &dir, 0);
      CheckMVdir(bmx - 1, bmy + 2, &dir, 1);
      CheckMVdir(bmx + 1, bmy + 2, &dir, 2);
      CheckMVdir(bmx + 2, bmy, &dir, 3);
      CheckMVdir(bmx + 1, bmy - 2, &dir, 4);
      CheckMVdir(bmx - 1, bmy - 2, &dir, 5);


      if (dir != -2)
      {
        bmx += hex2[dir + 1][0];
        bmy += hex2[dir + 1][1];
        /* half hexagon, not overlapping the previous iteration */
        for (int i = 1; i < i_me_range / 2 && IsVectorOK(bmx, bmy); i++)
        {
          const int odir = mod6m1[dir + 1];
          
          dir = -2;
          
          CheckMVdir(bmx + hex2[odir + 0][0], bmy + hex2[odir + 0][1], &dir, odir - 1);
          CheckMVdir(bmx + hex2[odir + 1][0], bmy + hex2[odir + 1][1], &dir, odir);
          CheckMVdir(bmx + hex2[odir + 2][0], bmy + hex2[odir + 2][1], &dir, odir + 1);
          if (dir == -2)
          {
            break;
          }
          bmx += hex2[dir + 1][0];
          bmy += hex2[dir + 1][1];
        }
      }

      bestMV.x = bmx;
      bestMV.y = bmy;
    }

    // square refine
    ExpandingSearch(1, 1, bmx, bmy);
  }

  void Refine()
  {
    // then, we refine, according to the search type
    switch (searchType) {
    case EXHAUSTIVE: {
      //		ExhaustiveSearch(nSearchParam);
      int mvx = bestMV.x;
      int mvy = bestMV.y;
      for (int i = 1; i <= nSearchParam; i++)// region is same as exhaustive, but ordered by radius (from near to far)
      {
        ExpandingSearch(i, 1, mvx, mvy);
      }
    }
    break;
    case HEX2SEARCH:
      Hex2Search(nSearchParam);
      break;
    default:
      // Not implemented
      break;
    }
  }

  int LumaSAD(const pixel_t *pRef0)
  {
    return SAD(pSrc[0], nSrcPitch[0], pRef0, nRefPitch[0]);
  }

  /* check if the vector (vx, vy) is better than the best vector found so far */
  template <bool isFirst>
  void	CheckMV(int vx, int vy)
  {		//here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
    if (IsVectorOK(vx, vy))
    {
      // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
      int cost = MotionDistorsion(vx, vy);
      if (cost >= nMinCost) return;

      typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64 >::type safe_sad_t;

      int sad = LumaSAD(GetRefBlock(vx, vy));
      cost += sad + (isFirst ? 0 : ((penaltyNew*(safe_sad_t)sad) >> 8));
      if (cost >= nMinCost) return;

      int saduv = SADCHROMA(pSrc[1], nSrcPitch[1], GetRefBlockU(vx, vy), nRefPitch[1])
        + SADCHROMA(pSrc[2], nSrcPitch[2], GetRefBlockV(vx, vy), nRefPitch[2]);
      cost += saduv + (isFirst ? 0 : ((penaltyNew*(safe_sad_t)saduv) >> 8));
      if (cost >= nMinCost) return;

      bestMV.x = vx;
      bestMV.y = vy;
      nMinCost = cost;
      bestMV.sad = sad + saduv;
    }
  }
  
  /* check if the vector (vx, vy) is better, and update dir accordingly, but not bestMV.x, y */
  void CheckMVdir(int vx, int vy, int *dir, int val)
  {
    if (IsVectorOK(vx, vy))
    {
      // from 2.5.11.9-SVP: no additional SAD calculations if partial sum is already above minCost
      int cost = MotionDistorsion(vx, vy);
      if (cost >= nMinCost) return;

      typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64 >::type safe_sad_t;

      int sad = LumaSAD(GetRefBlock(vx, vy));
      cost += sad + ((penaltyNew*(safe_sad_t)sad) >> 8);
      if (cost >= nMinCost) return;

      int saduv = SADCHROMA(pSrc[1], nSrcPitch[1], GetRefBlockU(vx, vy), nRefPitch[1])
        + SADCHROMA(pSrc[2], nSrcPitch[2], GetRefBlockV(vx, vy), nRefPitch[2]);
      cost += saduv + ((penaltyNew*(safe_sad_t)saduv) >> 8);
      if (cost >= nMinCost) return;

      nMinCost = cost;
      bestMV.sad = sad + saduv;
      *dir = val;
    }
  }

  void PseudoEPZSearch()
  {
    typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64 >::type safe_sad_t;
    FetchPredictors();

    // We treat zero alone
    // Do we bias zero with not taking into account distorsion ?
    bestMV.x = 0;
    bestMV.y = 0;
    int saduv = SADCHROMA(pSrc[1], nSrcPitch[1], GetRefBlockU(0, 0), nRefPitch[1])
      + SADCHROMA(pSrc[2], nSrcPitch[2], GetRefBlockV(0, 0), nRefPitch[2]);
    int sad = LumaSAD(GetRefBlock(0, 0));
    sad += saduv;
    bestMV.sad = sad;
    nMinCost = sad + ((penaltyZero*(safe_sad_t)sad) >> 8); // v.1.11.0.2

                                                                    // Global MV predictor  - added by Fizick
    globalMVPredictor = ClipMV(globalMVPredictor);
    //	if ( IsVectorOK(globalMVPredictor.x, globalMVPredictor.y ) )
    {
      saduv = SADCHROMA(pSrc[1], nSrcPitch[1], GetRefBlockU(globalMVPredictor.x, globalMVPredictor.y), nRefPitch[1])
        + SADCHROMA(pSrc[2], nSrcPitch[2], GetRefBlockV(globalMVPredictor.x, globalMVPredictor.y), nRefPitch[2]);
      sad = LumaSAD(GetRefBlock(globalMVPredictor.x, globalMVPredictor.y));
      sad += saduv;
      int cost = sad + ((pglobal*(safe_sad_t)sad) >> 8);

      if (cost < nMinCost)
      {
        bestMV.x = globalMVPredictor.x;
        bestMV.y = globalMVPredictor.y;
        bestMV.sad = sad;
        nMinCost = cost;
      }
      //	}
      //	Then, the predictor :
      //	if (   (( predictor.x != zeroMVfieldShifted.x ) || ( predictor.y != zeroMVfieldShifted.y ))
      //	    && (( predictor.x != globalMVPredictor.x ) || ( predictor.y != globalMVPredictor.y )))
      //	{
      saduv = SADCHROMA(pSrc[1], nSrcPitch[1], GetRefBlockU(predictor.x, predictor.y), nRefPitch[1])
        + SADCHROMA(pSrc[2], nSrcPitch[2], GetRefBlockV(predictor.x, predictor.y), nRefPitch[2]);
      sad = LumaSAD(GetRefBlock(predictor.x, predictor.y));
      sad += saduv;
      cost = sad;

      if (cost < nMinCost)
      {
        bestMV.x = predictor.x;
        bestMV.y = predictor.y;
        bestMV.sad = sad;
        nMinCost = cost;
      }
    }

    // then all the other predictors
    int npred = 4;

    for (int i = 0; i < npred; i++)
    {
      CheckMV<true>(predictors[i].x, predictors[i].y);
    }	// for i

      // then, we refine, according to the search type
    Refine();

      // we store the result
    vectors[blkIdx] = bestMV;
  }

public:
  PlaneOfBlocks(
    int _nBlkX, int _nBlkY, int _nBlkSizeX, int _nBlkSizeY, int nPel, int _nLevel,
    bool smallestPlane, bool chroma,
    int _nOverlapX, int _nOverlapY, int _xRatioUV, int _yRatioUV,
    int nPixelSize, int nBitsPerPixel, bool mt_flag, int chromaSADscale, 

    SearchType searchType, int nSearchParam, int PelSearch, int nLambda,
    int lsad, int penaltyNew, int plevel, bool global,
    int penaltyZero, int pglobal, int badSAD,
    int badrange, bool meander, bool tryMany,

    IScriptEnvironment* env)
    : nBlkX(_nBlkX)
    , nBlkY(_nBlkY)
    , nBlkSizeX(_nBlkSizeX)
    , nBlkSizeY(_nBlkSizeY)
    , nBlkCount(_nBlkX * _nBlkY)
    , nPel(nPel)
    , nLogPel(ilog2(nPel))	// nLogPel=0 for nPel=1, 1 for nPel=2, 2 for nPel=4, i.e. (x*nPel) = (x<<nLogPel)
    , nLogScale(_nLevel)
    , nScale(1 << _nLevel)
    , smallestPlane(smallestPlane)
    , chroma(chroma)
    , nOverlapX(_nOverlapX)
    , nOverlapY(_nOverlapY)
    , xRatioUV(_xRatioUV) // PF
    , nLogxRatioUV(ilog2(_xRatioUV))
    , yRatioUV(_yRatioUV)
    , nLogyRatioUV(ilog2(_yRatioUV))
    , nPixelSize(nPixelSize) // PF
    , nPixelShift(ilog2(nPixelSize)) // 161201
    , nBitsPerPixel(nBitsPerPixel) // PF
    , _mt_flag(mt_flag)
    , chromaSADscale(chromaSADscale)
    , searchType(searchType)
    , nSearchParam(nSearchParam)
    , vectors(nBlkCount)
    , PelSearch(PelSearch)
    , lsad(lsad)
    , penaltyNew(penaltyNew)
    , plevel(plevel)
    , global(global)
    , penaltyZero(penaltyZero)
    , pglobal(pglobal)
    , badSAD(badSAD)
    , badrange(badrange)
    , meander(meander)
    , tryMany(tryMany)
    , verybigSAD(3 * _nBlkSizeX * _nBlkSizeY * (nPixelSize == 4 ? 1 : (1 << nBitsPerPixel))) // * 256, pixelsize==2 -> 65536. Float:1
    , SAD(get_sad_function<pixel_t>(nBlkSizeX, nBlkSizeY, env))
    , SADCHROMA(get_sad_function<pixel_t>(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, env))
  {
    nLambdaLevel = nLambda / (nPel * nPel);
    if (plevel == 1)
    {
      nLambdaLevel *= nScale;	// scale lambda - Fizick
    }
    else if (plevel == 2)
    {
      nLambdaLevel *= nScale*nScale;
    }
  }

  /* search the vectors for the whole plane */
  void SearchMVs(KMFrame *pSrcFrame, KMFrame *pRefFrame, const VECTOR *_globalMV, MVData *out)
  {
    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Frame- and plane-related data preparation

    VECTOR globalMV = *_globalMV;
    globalMV.x *= nPel;
    globalMV.y *= nPel;

    // write the plane's header
    out->nCount = nBlkCount;

    pSrcYPlane = static_cast<KMPlane<pixel_t>*>(pSrcFrame->GetYPlane());
    pSrcUPlane = static_cast<KMPlane<pixel_t>*>(pSrcFrame->GetUPlane());
    pSrcVPlane = static_cast<KMPlane<pixel_t>*>(pSrcFrame->GetVPlane());
    pRefYPlane = static_cast<KMPlane<pixel_t>*>(pRefFrame->GetYPlane());
    pRefUPlane = static_cast<KMPlane<pixel_t>*>(pRefFrame->GetUPlane());
    pRefVPlane = static_cast<KMPlane<pixel_t>*>(pRefFrame->GetVPlane());

    nSrcPitch[0] = pSrcYPlane->GetPitch();
    if (chroma)
    {
      nSrcPitch[1] = pSrcUPlane->GetPitch();
      nSrcPitch[2] = pSrcVPlane->GetPitch();
    }
    nRefPitch[0] = pRefYPlane->GetPitch();
    if (chroma)
    {
      nRefPitch[1] = pRefUPlane->GetPitch();
      nRefPitch[2] = pRefVPlane->GetPitch();
    }

    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    VECTOR *pBlkData = out->data;

    y[0] = pSrcYPlane->GetVPadding();

    if (chroma)
    {
      y[1] = pSrcUPlane->GetVPadding();
      y[2] = pSrcVPlane->GetVPadding();
    }

    // Functions using float must not be used here

    int nBlkSizeX_Ovr[3] = { (nBlkSizeX - nOverlapX), (nBlkSizeX - nOverlapX) >> nLogxRatioUV, (nBlkSizeX - nOverlapX) >> nLogxRatioUV };
    int nBlkSizeY_Ovr[3] = { (nBlkSizeY - nOverlapY), (nBlkSizeY - nOverlapY) >> nLogyRatioUV, (nBlkSizeY - nOverlapY) >> nLogyRatioUV };

    for (blky = 0; blky < nBlkY; blky++)
    {
      blkScanDir = (blky % 2 == 0 || !meander) ? 1 : -1;
      // meander (alternate) scan blocks (even row left to right, odd row right to left)
      int blkxStart = (blky % 2 == 0 || !meander) ? 0 : nBlkX - 1;
      if (blkScanDir == 1) // start with leftmost block
      {
        x[0] = pSrcYPlane->GetHPadding();
        if (chroma)
        {
          x[1] = pSrcUPlane->GetHPadding();
          x[2] = pSrcVPlane->GetHPadding();
        }
      }
      else // start with rightmost block, but it is already set at prev row
      {
        x[0] = pSrcYPlane->GetHPadding() + nBlkSizeX_Ovr[0] * (nBlkX - 1);
        if (chroma)
        {
          x[1] = pSrcUPlane->GetHPadding() + nBlkSizeX_Ovr[1] * (nBlkX - 1);
          x[2] = pSrcVPlane->GetHPadding() + nBlkSizeX_Ovr[2] * (nBlkX - 1);
        }
      }

      for (int iblkx = 0; iblkx < nBlkX; iblkx++)
      {
        blkx = blkxStart + iblkx*blkScanDir;
        blkIdx = blky*nBlkX + blkx;
        iter = 0;
        //			DebugPrintf("BlkIdx = %d \n", blkIdx);

        // Resets the global predictor (it may have been clipped during the
        // previous block scan)
        globalMVPredictor = globalMV;

        pSrc[0] = pSrcYPlane->GetAbsolutePelPointer(x[0], y[0]);
        if (chroma)
        {
          pSrc[1] = pSrcUPlane->GetAbsolutePelPointer(x[1], y[1]);
          pSrc[2] = pSrcVPlane->GetAbsolutePelPointer(x[2], y[2]);
        }

        if (blky == 0)
        {
          nCurrentLambda = 0;
        }
        else
        {
          nCurrentLambda = nLambdaLevel;
        }

        // decreased padding of coarse levels
        int nHPaddingScaled = pSrcYPlane->GetHPadding() >> nLogScale;
        int nVPaddingScaled = pSrcYPlane->GetVPadding() >> nLogScale;
        /* computes search boundaries */
        nDxMax = nPel * (pSrcYPlane->GetExtendedWidth() - x[0] - nBlkSizeX - pSrcYPlane->GetHPadding() + nHPaddingScaled);
        nDyMax = nPel * (pSrcYPlane->GetExtendedHeight() - y[0] - nBlkSizeY - pSrcYPlane->GetVPadding() + nVPaddingScaled);
        nDxMin = -nPel * (x[0] - pSrcYPlane->GetHPadding() + nHPaddingScaled);
        nDyMin = -nPel * (y[0] - pSrcYPlane->GetVPadding() + nVPaddingScaled);

        /* search the mv */
        predictor = ClipMV(vectors[blkIdx]);
        
        // TODO: no need
        VECTOR zeroMV = { 0,0,0 };
        predictors[4] = ClipMV(zeroMV);

        PseudoEPZSearch();
        //			bestMV = zeroMV; // debug

        /* write the results */
        pBlkData[blkx] = bestMV;

        /* increment indexes & pointers */
        if (iblkx < nBlkX - 1)
        {
          x[0] += nBlkSizeX_Ovr[0] * blkScanDir;
          x[1] += nBlkSizeX_Ovr[1] * blkScanDir;
          x[2] += nBlkSizeX_Ovr[2] * blkScanDir;
        }
      }	// for iblkx

      pBlkData += nBlkX;

      y[0] += nBlkSizeY_Ovr[0];
      y[1] += nBlkSizeY_Ovr[1];
      y[2] += nBlkSizeY_Ovr[2];
    }	// for blky

    // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  }

  void EstimateGlobalMVDoubled(VECTOR *globalMVec)
  {
    std::vector<int> freq_arr(8192 * nPel * 2);

    for (int y = 0; y < 2; ++y)
    {
      const int      freqSize = int(freq_arr.size());
      memset(&freq_arr[0], 0, freqSize * sizeof(freq_arr[0])); // reset

      int            indmin = freqSize - 1;
      int            indmax = 0;

      // find most frequent x
      if (y == 0)
      {
        for (int i = 0; i < nBlkCount; i++)
        {
          int ind = (freqSize >> 1) + vectors[i].x;
          if (ind >= 0 && ind < freqSize)
          {
            ++freq_arr[ind];
            if (ind > indmax)
            {
              indmax = ind;
            }
            if (ind < indmin)
            {
              indmin = ind;
            }
          }
        }
      }

      // find most frequent y
      else
      {
        for (int i = 0; i < nBlkCount; i++)
        {
          int ind = (freqSize >> 1) + vectors[i].y;
          if (ind >= 0 && ind < freqSize)
          {
            ++freq_arr[ind];
            if (ind > indmax)
            {
              indmax = ind;
            }
            if (ind < indmin)
            {
              indmin = ind;
            }
          }
        }	// i < nBlkCount
      }

      int count = freq_arr[indmin];
      int index = indmin;
      for (int i = indmin + 1; i <= indmax; i++)
      {
        if (freq_arr[i] > count)
        {
          count = freq_arr[i];
          index = i;
        }
      }

      // most frequent value
      int result = index - (freqSize >> 1);
      if (y == 0) {
        globalMVec->x = result;
      }
      else {
        globalMVec->y = result;
      }
    }

    int medianx = globalMVec->x;
    int mediany = globalMVec->y;
    int meanvx = 0;
    int meanvy = 0;
    int num = 0;
    for (int i = 0; i < nBlkCount; i++)
    {
      if (abs(vectors[i].x - medianx) < 6
        && abs(vectors[i].y - mediany) < 6)
      {
        meanvx += vectors[i].x;
        meanvy += vectors[i].y;
        num += 1;
      }
    }

    // output vectors must be doubled for next (finer) scale level
    if (num > 0)
    {
      globalMVec->x = 2 * meanvx / num;
      globalMVec->y = 2 * meanvy / num;
    }
    else
    {
      globalMVec->x = 2 * medianx;
      globalMVec->y = 2 * mediany;
    }
  }

  void InterpolatePrediction(const PlaneOfBlocksBase* _pob)
  {
    const PlaneOfBlocks<pixel_t>& pob = *static_cast<const PlaneOfBlocks<pixel_t>*>(_pob);

    int normFactor = 3 - nLogPel + pob.nLogPel;
    int mulFactor = (normFactor < 0) ? -normFactor : 0;
    normFactor = (normFactor < 0) ? 0 : normFactor;
    int normov = (nBlkSizeX - nOverlapX)*(nBlkSizeY - nOverlapY);
    int aoddx = (nBlkSizeX * 3 - nOverlapX * 2);
    int aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
    int aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
    int aeveny = (nBlkSizeY * 3 - nOverlapY * 4);
    // note: overlapping is still (v2.5.7) not processed properly
    // PF todo make faster

    // 2.7.19.22 max safe: BlkX*BlkY: sqrt(2147483647 / 3 / 255) = 1675 ,(2147483647 = 0x7FFFFFFF)
    bool bNoOverlap = (nOverlapX == 0 && nOverlapY == 0);
    bool isSafeBlkSizeFor8bits = (nBlkSizeX*nBlkSizeY) < 1675;
    bool bSmallOverlap = nOverlapX <= (nBlkSizeX >> 1) && nOverlapY <= (nBlkSizeY >> 1);

    for (int l = 0, index = 0; l < nBlkY; l++)
    {
      for (int k = 0; k < nBlkX; k++, index++)
      {
        VECTOR v1, v2, v3, v4;
        int i = k;
        int j = l;
        if (i >= 2 * pob.nBlkX)
        {
          i = 2 * pob.nBlkX - 1;
        }
        if (j >= 2 * pob.nBlkY)
        {
          j = 2 * pob.nBlkY - 1;
        }
        int offy = -1 + 2 * (j % 2);
        int offx = -1 + 2 * (i % 2);
        int iper2 = i / 2;
        int jper2 = j / 2;

        if ((i == 0) || (i >= 2 * pob.nBlkX - 1))
        {
          if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
          {
            v1 = v2 = v3 = v4 = pob.vectors[iper2 + (jper2)* pob.nBlkX];
          }
          else
          {
            v1 = v2 = pob.vectors[iper2 + (jper2)* pob.nBlkX];
            v3 = v4 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
          }
        }
        else if ((j == 0) || (j >= 2 * pob.nBlkY - 1))
        {
          v1 = v2 = pob.vectors[iper2 + (jper2)* pob.nBlkX];
          v3 = v4 = pob.vectors[iper2 + offx + (jper2)* pob.nBlkX];
        }
        else
        {
          v1 = pob.vectors[iper2 + (jper2)* pob.nBlkX];
          v2 = pob.vectors[iper2 + offx + (jper2)* pob.nBlkX];
          v3 = pob.vectors[iper2 + (jper2 + offy) * pob.nBlkX];
          v4 = pob.vectors[iper2 + offx + (jper2 + offy) * pob.nBlkX];
        }
        typedef typename std::conditional < sizeof(pixel_t) == 1, int, __int64 >::type safe_sad_t;
        safe_sad_t tmp_sad; // 16 bit worst case: 16 * sad_max: 16 * 3x32x32x65536 = 4+5+5+16 > 2^31 over limit
                            // in case of BlockSize > 32, e.g. 128x128x65536 is even more: 7+7+16=30 bits

        if (bNoOverlap)
        {
          vectors[index].x = 9 * v1.x + 3 * v2.x + 3 * v3.x + v4.x;
          vectors[index].y = 9 * v1.y + 3 * v2.y + 3 * v3.y + v4.y;
          tmp_sad = 9 * (safe_sad_t)v1.sad + 3 * (safe_sad_t)v2.sad + 3 * (safe_sad_t)v3.sad + (safe_sad_t)v4.sad + 8;

        }
        else if (bSmallOverlap) // corrected in v1.4.11
        {
          int	ax1 = (offx > 0) ? aoddx : aevenx;
          int ax2 = (nBlkSizeX - nOverlapX) * 4 - ax1;
          int ay1 = (offy > 0) ? aoddy : aeveny;
          int ay2 = (nBlkSizeY - nOverlapY) * 4 - ay1;
          int a11 = ax1*ay1, a12 = ax1*ay2, a21 = ax2*ay1, a22 = ax2*ay2;
          vectors[index].x = (a11*v1.x + a21*v2.x + a12*v3.x + a22*v4.x) / normov;
          vectors[index].y = (a11*v1.y + a21*v2.y + a12*v3.y + a22*v4.y) / normov;
          if (isSafeBlkSizeFor8bits && sizeof(pixel_t) == 1) {
            // old max blkSize==32 worst case: 
            //   normov = (32-2)*(32-2) 
            //   sad = 32x32x255 *3 (3 planes) // 705,024,000 < 2^31 OK
            // blkSize == 48 worst case:
            //   normov = (48-2)*(48-2) = 2116
            //   sad = 48x48x255 * 3 // 3,729,576,960 not OK, already fails in 8 bits
            // max safe: BlkX*BlkY: sqrt(0x7FFFFFF / 3 / 255) = 1675
            tmp_sad = ((safe_sad_t)a11*v1.sad + (safe_sad_t)a21*v2.sad + (safe_sad_t)a12*v3.sad + (safe_sad_t)a22*v4.sad) / normov;
          }
          else {
            // safe multiplication
            tmp_sad = ((__int64)a11*v1.sad + (__int64)a21*v2.sad + (__int64)a12*v3.sad + (__int64)a22*v4.sad) / normov;
          }
        }
        else // large overlap. Weights are not quite correct but let it be
        {
          vectors[index].x = (v1.x + v2.x + v3.x + v4.x) << 2;
          vectors[index].y = (v1.y + v2.y + v3.y + v4.y) << 2;
          tmp_sad = ((safe_sad_t)v1.sad + v2.sad + v3.sad + v4.sad + 2) << 2;
        }
        vectors[index].x = (vectors[index].x >> normFactor) << mulFactor;
        vectors[index].y = (vectors[index].y >> normFactor) << mulFactor;
        vectors[index].sad = (int)(tmp_sad >> 4);
      }	// for k < nBlkX
    }	// for l < nBlkY
  }

    // returns length (in bytes) written
  int WriteDefault(MVData *out)
  {
    out->nCount = nBlkCount;
    for (int i = 0; i < nBlkCount; ++i)
    {
      out->data[i].x = 0;
      out->data[i].y = 0;
      out->data[i].sad = verybigSAD; // float or int!!
    }
    return GetArraySize();
  }

  int GetArraySize()
  {
    return offsetof(MVData, data) + nBlkCount * sizeof(VECTOR);
  }
};

class GroupOfPlanes
{
  const int nLevelCount;
  const bool global;

  std::unique_ptr<std::unique_ptr<PlaneOfBlocksBase>[]> planes;

public:
  GroupOfPlanes(
    int nBlkSizeX, int nBlkSizeY, int nLevelCount, int nPel, bool chroma,
    int nOverlapX, int nOverlapY, const LevelInfo* linfo, int xRatioUV, int yRatioUV,
    int divideExtra, int nPixelSize, int nBitsPerPixel,
    bool mt_flag, int chromaSADScale, 
    
    SearchType searchType, int nSearchParam, int nPelSearch, int nLambda,
    int lsad, int pnew, int plevel, bool global,
    int penaltyZero, int pglobal, int badSAD,
    int badrange, bool meander, bool tryMany,

    IScriptEnvironment *env)
    : nLevelCount(nLevelCount)
    , global(global)
    , planes(new std::unique_ptr<PlaneOfBlocksBase>[nLevelCount])
  {
    for (int i = 0; i < nLevelCount; i++) {
      int nPelLevel = (i == 0) ? nPel : 1;
      int nBlkX = linfo[i].nBlkX;
      int nBlkY = linfo[i].nBlkY;

      // full search for coarse planes
      SearchType searchTypeLevel =
        (i == 0 || searchType == HSEARCH || searchType == VSEARCH)
        ? searchType
        : EXHAUSTIVE;

      // special case for finest level
      int nSearchParamLevel = (i == 0) ? nPelSearch : nSearchParam;

      if (nPixelSize == 1) {
        planes[i] = std::unique_ptr<PlaneOfBlocksBase>(
          new PlaneOfBlocks<uint8_t>(nBlkX, nBlkY, nBlkSizeX, nBlkSizeY, nPelLevel, i,
            i == nLevelCount - 1, chroma, nOverlapX, nOverlapY, xRatioUV, yRatioUV,
            nPixelSize, nBitsPerPixel, mt_flag, chromaSADScale,

            searchTypeLevel, nSearchParam, nSearchParamLevel, nLambda, lsad, pnew,
            plevel, global, penaltyZero, pglobal, badSAD, badrange, meander,
            tryMany,

            env));
      }
      else {
        planes[i] = std::unique_ptr<PlaneOfBlocksBase>(
          new PlaneOfBlocks<uint16_t>(nBlkX, nBlkY, nBlkSizeX, nBlkSizeY, nPelLevel, i,
            i == nLevelCount - 1, chroma, nOverlapX, nOverlapY, xRatioUV, yRatioUV,
            nPixelSize, nBitsPerPixel, mt_flag, chromaSADScale,

            searchTypeLevel, nSearchParam, nSearchParamLevel, nLambda, lsad, pnew,
            plevel, global, penaltyZero, pglobal, badSAD, badrange, meander,
            tryMany,

            env));
      }
    }
  }

  void SearchMVs(KMSuperFrame *pSrcGOF, KMSuperFrame *pRefGOF, MVDataGroup *out)
  {
    out->isValid = true;

    BYTE* ptr = out->data;

    // create and init global motion vector as zero
    VECTOR globalMV = { 0, 0, -1 };

    // Refining the search until we reach the highest detail interpolation.
    for (int i = nLevelCount - 1; i >= 0; i--) 
    {
      if (i != nLevelCount - 1) {
        if (global)
        {
          // get updated global MV (doubled)
          planes[i + 1]->EstimateGlobalMVDoubled(&globalMV);
        }

        planes[i]->InterpolatePrediction(planes[i + 1].get());
      }

      //		DebugPrintf("SearchMV level %i", i);
      planes[i]->SearchMVs(
        pSrcGOF->GetFrame(i),
        pRefGOF->GetFrame(i),
        &globalMV,
        reinterpret_cast<MVData*>(ptr)
      );

      ptr += planes[i]->GetArraySize();
    }
  }

  void WriteDefault(MVDataGroup *out)
  {
    out->isValid = false;

    BYTE* ptr = out->data;

    // write planes
    for (int i = nLevelCount - 1; i >= 0; i--)
    {
      ptr += planes[i]->WriteDefault(reinterpret_cast<MVData*>(ptr));
    }
  }

  int GetArraySize()
  {
    int size = offsetof(MVDataGroup, data);
    for (int i = nLevelCount - 1; i >= 0; i--)
    {
      size += planes[i]->GetArraySize();
    }
    return size;
  }
};


class KMAnalyse : public GenericVideoFilter
{
protected:
  KMVParam params;
  std::unique_ptr<GroupOfPlanes> _vectorfields_aptr;
  std::unique_ptr<KMSuperFrame> pSrcSF, pRefSF;

private:

  void LoadSourceFrame(KMSuperFrame *sf, PVideoFrame &src)
  {
    const unsigned char *	pSrcY;
    const unsigned char *	pSrcU;
    const unsigned char *	pSrcV;
    int				nSrcPitchY;
    int				nSrcPitchUV;

    pSrcY = src->GetReadPtr(PLANAR_Y);
    pSrcU = src->GetReadPtr(PLANAR_U);
    pSrcV = src->GetReadPtr(PLANAR_V);
    nSrcPitchY = src->GetPitch(PLANAR_Y) >> params.nPixelShift;
    nSrcPitchUV = src->GetPitch(PLANAR_U) >> params.nPixelShift;

    sf->SetTarget(
      (BYTE*)pSrcY, nSrcPitchY,
      (BYTE*)pSrcU, nSrcPitchUV,
      (BYTE*)pSrcV, nSrcPitchUV
    ); // v2.0
  }
public:

  KMAnalyse(
    PClip child, int blksizex, int blksizey, int lv, int st, int stp,
    int pelSearch, bool isb, int lambda, bool chroma, int df, int lsad,
    int plevel, bool global, int pnew, int penaltyZero, int pglobal,
    int overlapx, int overlapy, const char* _outfilename, int dctmode,
    int divide, int sadx264, int badSAD, int badrange, bool isse,
    bool meander, bool temporal_flag, bool tryMany, bool multi_flag,
    bool mt_flag, int _chromaSADScale, IScriptEnvironment* env)
    : GenericVideoFilter(child)
    , params(KMVParam::MV_FRAME)
    , _vectorfields_aptr()
  {

    int nPixelSize = vi.ComponentSize();
    int nBitsPerPixel = vi.BitsPerComponent();

    if (nPixelSize == 4)
    {
      env->ThrowError("KMAnalyse: Clip with float pixel type is not supported");
    }
    if (!vi.IsYUV())
    {
      env->ThrowError("KMAnalyse: Clip must be YUV");
    }

    params = *KMVParam::GetParam(vi, env);

    params.nDataType = KMVParam::MV_FRAME;
    params.chromaSADScale = _chromaSADScale;
    params.nBlkSizeX = blksizex;
    params.nBlkSizeY = blksizey;
    params.nOverlapX = overlapx;
    params.nOverlapY = overlapy;
    params.nDeltaFrame = df;

    const std::vector< std::pair< int, int > > allowed_blksizes = { { 32,32 },{ 16,16 } };
    bool found = false;
    for (int i = 0; i < allowed_blksizes.size(); i++) {
      if (params.nBlkSizeX == allowed_blksizes[i].first && params.nBlkSizeY == allowed_blksizes[i].second) {
        found = true;
        break;
      }
    }
    if (!found) {
      env->ThrowError(
        "KMAnalyse: Invalid block size: %d x %d", params.nBlkSizeX, params.nBlkSizeY);
    }
    if (params.nPel != 1
      && params.nPel != 2)
    {
      env->ThrowError("KMAnalyse: pel has to be 1 or 2 or 4");
    }
    if (overlapx < 0 || overlapx > blksizex / 2
      || overlapy < 0 || overlapy > blksizey / 2)
    {
      env->ThrowError("KMAnalyse: overlap must be less or equal than half block size");
    }
    if (overlapx % params.xRatioUV || overlapy % params.yRatioUV) // PF subsampling-aware
    {
      env->ThrowError("KMAnalyse: wrong overlap for the colorspace subsampling");
    }
    if (vi.IsY())
    {
      chroma = false;
    }

    const int nBlkX = (params.nWidth - params.nOverlapX) / (params.nBlkSizeX - params.nOverlapX);
    const int nBlkY = (params.nHeight - params.nOverlapY) / (params.nBlkSizeY - params.nOverlapY);
    const int nWidth_B = (params.nBlkSizeX - params.nOverlapX) * nBlkX + params.nOverlapX; // covered by blocks
    const int nHeight_B = (params.nBlkSizeY - params.nOverlapY) * nBlkY + params.nOverlapY;

    // calculate valid levels
    int				nLevelsMax = 0;
    // at last one block
    while (((nWidth_B >> nLevelsMax) - params.nOverlapX) / (params.nBlkSizeX - params.nOverlapX) > 0
      && ((nHeight_B >> nLevelsMax) - params.nOverlapY) / (params.nBlkSizeY - params.nOverlapY) > 0)
    {
      ++nLevelsMax;
    }

    int nAnalyzeLevel = (lv > 0) ? lv : nLevelsMax + lv;
    if (nAnalyzeLevel > params.nLevels)
    {
      env->ThrowError(
        "MAnalyse: it is not enough levels  in super clip (%d), "
        "while MAnalyse asks %d", params.nLevels, nAnalyzeLevel
      );
    }
    if (nAnalyzeLevel < 1
      || nAnalyzeLevel > nLevelsMax)
    {
      env->ThrowError(
        "MAnalyse: non-valid number of levels (%d)", nAnalyzeLevel
      );
    }

    params.isBackward = isb;

    // 各階層のブロック数を計算
    params.levelInfo.clear();
    for (int i = 0; i < nAnalyzeLevel; i++) {
      LevelInfo linfo = {
        ((nWidth_B >> i) - params.nOverlapX) / (params.nBlkSizeX - params.nOverlapX),
        ((nHeight_B >> i) - params.nOverlapY) / (params.nBlkSizeY - params.nOverlapY)
      };
      params.levelInfo.push_back(linfo);
    }

    KMVParam::SetParam(vi, &params);

    lsad = lsad * (blksizex * blksizey) / 64 * (1 << (params.nBitsPerPixel - 8)); // normalized to 8x8 blocksize todo: float
    badSAD = badSAD * (blksizex * blksizey) / 64 * (1 << (params.nBitsPerPixel - 8));

    // not below value of 0 at finest level
    pelSearch = (pelSearch <= 0) ? params.nPel : pelSearch;

    SearchType searchType;
    int nSearchParam;

    switch (st)
    {
    case 0:
      searchType = ONETIME;
      nSearchParam = (stp < 1) ? 1 : stp;
      break;
    case 1:
      searchType = NSTEP;
      nSearchParam = (stp < 0) ? 0 : stp;
      break;
    case 3:
      searchType = EXHAUSTIVE;
      nSearchParam = (stp < 1) ? 1 : stp;
      break;
    case 4:
      searchType = HEX2SEARCH;
      nSearchParam = (stp < 1) ? 1 : stp;
      break;
    case 5:
      searchType = UMHSEARCH;
      nSearchParam = (stp < 1) ? 1 : stp; // really min is 4
      break;
    case 6:
      searchType = HSEARCH;
      nSearchParam = (stp < 1) ? 1 : stp;
      break;
    case 7:
      searchType = VSEARCH;
      nSearchParam = (stp < 1) ? 1 : stp;
      break;
    case 2:
    default:
      searchType = LOGARITHMIC;
      nSearchParam = (stp < 1) ? 1 : stp;
    }

    pSrcSF = std::unique_ptr<KMSuperFrame>(new KMSuperFrame(&params));
    pRefSF = std::unique_ptr<KMSuperFrame>(new KMSuperFrame(&params));

    _vectorfields_aptr = std::unique_ptr<GroupOfPlanes>(new GroupOfPlanes(
      params.nBlkSizeX,
      params.nBlkSizeY,
      int(params.levelInfo.size()),
      params.nPel,
      params.chroma,
      params.nOverlapX,
      params.nOverlapY,
      params.levelInfo.data(),
      params.xRatioUV, // PF
      params.yRatioUV,
      0,
      params.nPixelSize, // PF
      params.nBitsPerPixel,
      false,
      params.chromaSADScale,

      searchType,
      nSearchParam,
      pelSearch,
      lambda,
      lsad, pnew,
      plevel,
      global,
      penaltyZero,
      pglobal,
      badSAD,
      badrange,
      meander,
      tryMany,

      env
    ));

    // Defines the format of the output vector clip
    const int		out_frame_bytes = _vectorfields_aptr->GetArraySize();
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 2048;
    vi.height = nblock(out_frame_bytes, vi.width * 4);
  }

  PVideoFrame __stdcall	GetFrame(int n, ::IScriptEnvironment* env)
  {
    const int nsrc = n;
    const int nbr_src_frames = child->GetVideoInfo().num_frames;
    const int offset = (params.isBackward) ? params.nDeltaFrame : -params.nDeltaFrame;
    const int minframe = std::max(-offset, 0);
    const int maxframe = nbr_src_frames + std::min(-offset, 0);
    const int nref = nsrc + offset;

    PVideoFrame dst = env->NewVideoFrame(vi);
    MVDataGroup* pDst = reinterpret_cast<MVDataGroup*>(dst->GetWritePtr());

    // 0: 2_size_validity+(foreachblock(1_validity+blockCount*3))

    if (nsrc < minframe || nsrc >= maxframe)
    {
      // fill all vectors with invalid data
      _vectorfields_aptr->WriteDefault(pDst);
    }
    else
    {
      //		DebugPrintf ("MVAnalyse: Get src frame %d",nsrc);
      ::PVideoFrame	src = child->GetFrame(nsrc, env); // v2.0
      LoadSourceFrame(pSrcSF.get(), src);

      //		DebugPrintf ("MVAnalyse: Get ref frame %d", nref);
      //		DebugPrintf ("MVAnalyse frame %i backward=%i", nsrc, srd._analysis_data.isBackward);
      ::PVideoFrame	ref = child->GetFrame(nref, env); // v2.0
      LoadSourceFrame(pRefSF.get(), ref);

      _vectorfields_aptr->SearchMVs(pSrcSF.get(), pRefSF.get(), pDst);
    }

    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    int blksize = args[1].AsInt(16);       // block size horizontal
    int blksizeV = blksize; // block size vertical

    int lambda;
    int lsad;
    int pnew;
    int plevel;
    bool global;
    int overlap = args[2].AsInt(8);

    bool truemotion = false; // preset added in v0.9.13
    lambda = args[6].AsInt(400);
    lsad = args[7].AsInt(400);
    pnew = 25;
    plevel = 0;
    global = args[8].AsBool(true);

    return new KMAnalyse(
      args[0].AsClip(),       // super
      blksize,
      blksizeV,                // v.1.7
      0,       // levels skip
      4,       // search type
      2,       // search parameter
      2,       // search parameter at finest level
      args[3].AsBool(false),  // is backward
      lambda,                  // lambda
      args[4].AsBool(true),   // chroma = true since v1.0.1
      args[5].AsInt(1),       // delta frame
      lsad,                    // lsad - SAD limit for lambda using - added by Fizick (was internal fixed to 400 since v0.9.7)
      plevel,                  // plevel - exponent for lambda scaling on level size  - added by Fizick
      global,                  // use global motion predictor - added by Fizick
      pnew,                    // pnew - cost penalty for new candidate vector - added by Fizick
      pnew,    // pzero - v1.10.3
      0,       // pglobal
      overlap,                 // overlap horizontal
      overlap, // overlap vertical - v1.7
      "",   // outfile - v1.2.6
      0,       // dct
      0,       // divide
      0,       // sadx264
      10000,   // badSAD
      24,      // badrange
      true,   // isse
      args[9].AsBool(true),   // meander blocks scan
      false,  // temporal predictor
      false,  // try many
      false,  // multi
      false,  // mt
      0,   // scaleCSAD
      env
    );
  }
};

class KMSuperCheck : public GenericVideoFilter
{
  PClip kmsuper;
  PClip mvsuper;

  const KMVParam* params;

  std::unique_ptr<KMSuperFrame> pKSF;
  std::unique_ptr<KMSuperFrame> pMSF;

  void LoadSourceFrame(KMSuperFrame *sf, PVideoFrame &src)
  {
    const unsigned char *	pSrcY;
    const unsigned char *	pSrcU;
    const unsigned char *	pSrcV;
    int				nSrcPitchY;
    int				nSrcPitchUV;

    pSrcY = src->GetReadPtr(PLANAR_Y);
    pSrcU = src->GetReadPtr(PLANAR_U);
    pSrcV = src->GetReadPtr(PLANAR_V);
    nSrcPitchY = src->GetPitch(PLANAR_Y) >> params->nPixelShift;
    nSrcPitchUV = src->GetPitch(PLANAR_U) >> params->nPixelShift;

    sf->SetTarget(
      (BYTE*)pSrcY, nSrcPitchY,
      (BYTE*)pSrcU, nSrcPitchUV,
      (BYTE*)pSrcV, nSrcPitchUV
    ); // v2.0
  }

  template <typename pixel_t>
  void CheckPlane(const KMPlane<pixel_t>* kPlane, const KMPlane<pixel_t>* mPlane, IScriptEnvironment* env)
  {
    int nPel = kPlane->GetNPel();
    int nPlanes = nPel * nPel;
    int nPitch = kPlane->GetPitch();
    int w = kPlane->GetExtendedWidth();
    int h = kPlane->GetExtendedHeight();

    // サブピクセルループ
    for (int sy = 0; sy < nPel; ++sy) {
      for (int sx = 0; sx < nPel; ++sx) {
        const pixel_t* kptr = kPlane->GetAbsolutePointer(sx, sy);
        const pixel_t* mptr = mPlane->GetAbsolutePointer(sx, sy);
        // 画素ループ
        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            pixel_t kv = kptr[x + y * nPitch];
            pixel_t mv = mptr[x + y * nPitch];
            if (std::abs(kv - mv) > 1) {
              env->ThrowError("ERROR !!!");
            }
          }
        }
      }
    }

  }

  void CheckPlane(const KMPlaneBase* kPlane, const KMPlaneBase* mPlane, IScriptEnvironment* env)
  {
    if (params->nPixelSize == 1) {
      CheckPlane<uint8_t>(static_cast<const KMPlane<uint8_t>*>(kPlane), static_cast<const KMPlane<uint8_t>*>(mPlane), env);
    }
    else {
      CheckPlane<uint16_t>(static_cast<const KMPlane<uint16_t>*>(kPlane), static_cast<const KMPlane<uint16_t>*>(mPlane), env);
    }
  }

public:
  KMSuperCheck(PClip kmsuper, PClip mvsuper, PClip view, IScriptEnvironment* env)
    : GenericVideoFilter(view)
    , kmsuper(kmsuper)
    , mvsuper(mvsuper)
    , params(KMVParam::GetParam(kmsuper->GetVideoInfo(), env))
    , pKSF(new KMSuperFrame(params))
    , pMSF(new KMSuperFrame(params))
  {
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame ret = child->GetFrame(n, env);
    PVideoFrame ksuper = kmsuper->GetFrame(n, env);
    PVideoFrame msuper = mvsuper->GetFrame(n, env);

    LoadSourceFrame(pKSF.get(), ksuper);
    LoadSourceFrame(pMSF.get(), msuper);

    for (int i = 0; i < params->nLevels; ++i) {
      KMFrame* kframe = pKSF->GetFrame(i);
      KMFrame* mframe = pMSF->GetFrame(i);

      CheckPlane(kframe->GetYPlane(), mframe->GetYPlane(), env);
      if (params->chroma) {
        CheckPlane(kframe->GetUPlane(), mframe->GetUPlane(), env);
        CheckPlane(kframe->GetVPlane(), mframe->GetVPlane(), env);
      }
    }

    return ret;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KMSuperCheck(args[0].AsClip(), args[1].AsClip(), args[2].AsClip(), env);
  }
};

class KMAnalyzeCheck : public GenericVideoFilter
{
  PClip kmv;
  PClip mvv;

  const KMVParam* params;

  template <typename pixel_t>
  void CheckPlane(const KMPlane<pixel_t>* kPlane, const KMPlane<pixel_t>* mPlane)
  {
    int nPel = kPlane->GetNPel();
    int nPlanes = nPel * nPel;
    int nPitch = kPlane->GetPitch();
    int w = kPlane->GetExtendedWidth();
    int h = kPlane->GetExtendedHeight();

    // サブピクセルループ
    for (int sy = 0; sy < nPel; ++sy) {
      for (int sx = 0; sx < nPel; ++sx) {
        const pixel_t* kptr = kPlane->GetAbsolutePointer(sx, sy);
        const pixel_t* mptr = mPlane->GetAbsolutePointer(sx, sy);
        // 画素ループ
        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            pixel_t kv = kptr[x + y * nPitch];
            pixel_t mv = mptr[x + y * nPitch];
            if (std::abs(kv - mv) > 1) {
              //
            }
          }
        }
      }
    }

  }

  void CheckPlane(const KMPlaneBase* kPlane, const KMPlaneBase* mPlane)
  {
    if (params->nPixelSize == 1) {
      CheckPlane<uint8_t>(static_cast<const KMPlane<uint8_t>*>(kPlane), static_cast<const KMPlane<uint8_t>*>(mPlane));
    }
    else {
      CheckPlane<uint16_t>(static_cast<const KMPlane<uint16_t>*>(kPlane), static_cast<const KMPlane<uint16_t>*>(mPlane));
    }
  }

  void GetMVData(int n, const int*& pMv, int& data_size, IScriptEnvironment* env)
  {
    VideoInfo vi = mvv->GetVideoInfo();
    PVideoFrame mmvframe = mvv->GetFrame(n, env);

    // vector clip is rgb32
    const int		bytes_per_pix = vi.BitsPerPixel() >> 3;
    // for calculation of buffer size 

    const int		line_size = vi.width * bytes_per_pix;	// in bytes
    data_size = vi.height * line_size / sizeof(int);	// in 32-bit words

    pMv = reinterpret_cast<const int*>(mmvframe->GetReadPtr());
    int header_size = pMv[0];
    int nMagicKey1 = pMv[1];
    if (nMagicKey1 != 0x564D)
    {
      env->ThrowError("MVTools: invalid vector stream");
    }
    int nVersion1 = pMv[2];
    if (nVersion1 != 5)
    {
      env->ThrowError("MVTools: incompatible version of vector stream");
    }

    // 17.05.22 filling from motion vector clip
    const int		hs_i32 = header_size / sizeof(int);
    pMv += hs_i32;									// go to data - v1.8.1
    data_size -= hs_i32;
  }

public:
  KMAnalyzeCheck(PClip kmv, PClip mvv, PClip view, IScriptEnvironment* env)
    : GenericVideoFilter(view)
    , kmv(kmv)
    , mvv(mvv)
    , params(KMVParam::GetParam(kmv->GetVideoInfo(), env))
  {
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    PVideoFrame ret = child->GetFrame(n, env);
    PVideoFrame kmvframe = kmv->GetFrame(n, env);

    const MVDataGroup* kdata = reinterpret_cast<const MVDataGroup*>(kmvframe->GetReadPtr());
    const MVData* kptr = reinterpret_cast<const MVData*>(kdata->data);

    const int* pMv;
    int data_size;
    GetMVData(n, pMv, data_size, env);

    // validity
    if (kdata->isValid != pMv[1]) {
      env->ThrowError("Validity missmatch");
    }
    pMv += 2;

    for (int i = int(params->levelInfo.size()) - 1; i >= 0; i--) {
      int nBlkCount = params->levelInfo[i].nBlkX * params->levelInfo[i].nBlkY;
      int length = pMv[0];
      
      for (int v = 0; v < nBlkCount; ++v) {
        VECTOR kmv = kptr->data[v];
        VECTOR mmv = { pMv[v * 3 + 1], pMv[v * 3 + 2], pMv[v * 3 + 3] };
        if (kmv.x != mmv.x || kmv.y != mmv.y || kmv.sad != mmv.sad) {
          env->ThrowError("Motion vector missmatch");
        }
      }

      pMv += length;
      kptr = reinterpret_cast<const MVData*>(&kptr->data[kptr->nCount]);
    }

    return ret;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KMAnalyzeCheck(args[0].AsClip(), args[1].AsClip(), args[2].AsClip(), env);
  }
};

void AddFuncMV(IScriptEnvironment* env)
{
  env->AddFunction("KMSuper", "c[debug]i", KMSuper::Create, 0);
  env->AddFunction("KMAnalyse",
    "c[blksize]i[overlap]i[isb]b[chroma]b[delta]i[lambda]i[lsad]i[global]b[meander]b",
    KMAnalyse::Create, 0);

  env->AddFunction("KMSuperCheck", "[kmsuper]c[mvsuper]c[view]c", KMSuperCheck::Create, 0);
  env->AddFunction("KMAnalyzeCheck", "[kmanalyze]c[mvanalyze]c[view]c", KMAnalyzeCheck::Create, 0);
}
