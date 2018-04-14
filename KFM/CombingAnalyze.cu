
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"


__device__ __host__ void CountFlag(int cnt[3], int flag)
{
  if (flag & MOVE) cnt[0]++;
  if (flag & SHIMA) cnt[1]++;
  if (flag & LSHIMA) cnt[2]++;
}

void cpu_count_fmflags(FMCount* dst, const uchar4* flagp, int width, int height, int pitch)
{
  int cnt[3] = { 0 };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uchar4 flags = flagp[x + y * pitch];
      CountFlag(cnt, flags.x);
      CountFlag(cnt, flags.y);
      CountFlag(cnt, flags.z);
      CountFlag(cnt, flags.w);
    }
  }
  dst->move = cnt[0];
  dst->shima = cnt[1];
  dst->lshima = cnt[2];
}

__global__ void kl_init_fmcount(FMCount* dst)
{
  int tx = threadIdx.x;
  dst[tx].move = dst[tx].shima = dst[tx].lshima = 0;
}

enum {
  FM_COUNT_TH_W = 32,
  FM_COUNT_TH_H = 16,
  FM_COUNT_THREADS = FM_COUNT_TH_W * FM_COUNT_TH_H,
};

__global__ void kl_count_fmflags(FMCount* dst, const uchar4* flagp, int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * FM_COUNT_TH_W;
  int y = threadIdx.y + blockIdx.y * FM_COUNT_TH_H;
  int tid = threadIdx.x + threadIdx.y * FM_COUNT_TH_W;

  int cnt[3] = { 0 };

  if (x < width && y < height) {
    uchar4 flags = flagp[x + y * pitch];
    CountFlag(cnt, flags.x);
    CountFlag(cnt, flags.y);
    CountFlag(cnt, flags.z);
    CountFlag(cnt, flags.w);
  }

  __shared__ int sbuf[FM_COUNT_THREADS * 3];
  dev_reduceN<int, 3, FM_COUNT_THREADS, AddReducer<int>>(tid, cnt, sbuf);

  if (tid == 0) {
    atomicAdd(&dst->move, cnt[0]);
    atomicAdd(&dst->shima, cnt[1]);
    atomicAdd(&dst->lshima, cnt[2]);
  }
}

class KFMFrameAnalyze : public KFMFilterBase
{
  VideoInfo padvi;
  VideoInfo flagvi;

  FrameAnalyzeParam prmY;
  FrameAnalyzeParam prmC;

  void CountFlags(PVideoFrame& flag, PVideoFrame& dst, int parity, PNeoEnv env)
  {
    const uchar4* flagp = reinterpret_cast<const uchar4*>(flag->GetReadPtr(PLANAR_Y));
    FMCount* fmcnt = reinterpret_cast<FMCount*>(dst->GetWritePtr());
    int width4 = srcvi.width >> 2;
    int flagPitch = flag->GetPitch(PLANAR_Y) / sizeof(uchar4);

    FMCount* fmcnt0 = &fmcnt[0];
    FMCount* fmcnt1 = &fmcnt[1];
    if (!parity) {
      std::swap(fmcnt0, fmcnt1);
    }

    if (IS_CUDA) {
      dim3 threads(FM_COUNT_TH_W, FM_COUNT_TH_H);
      dim3 blocks(nblocks(srcvi.width, threads.x), nblocks(srcvi.height / 2, threads.y));
      kl_init_fmcount << <1, 2 >> > (fmcnt);
      DEBUG_SYNC;
      kl_count_fmflags << <blocks, threads >> >(
        fmcnt0, flagp, width4, srcvi.height / 2, flagPitch * 2);
      DEBUG_SYNC;
      kl_count_fmflags << <blocks, threads >> >(
        fmcnt1, flagp + flagPitch, width4, srcvi.height / 2, flagPitch * 2);
      DEBUG_SYNC;
    }
    else {
      cpu_count_fmflags(fmcnt0, flagp, width4, srcvi.height / 2, flagPitch * 2);
      cpu_count_fmflags(fmcnt1, flagp + flagPitch, width4, srcvi.height / 2, flagPitch * 2);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    int parity = child->GetParity(n);
    PVideoFrame f0 = child->GetFrame(n, env);
    PVideoFrame f1 = child->GetFrame(n + 1, env);
    PVideoFrame f0padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame f1padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame fflag = env->NewVideoFrame(flagvi);
    PVideoFrame dst = env->NewVideoFrame(vi);

    // TODO: 切り出し
    CopyFrame<pixel_t>(f0, f0padded, env);
    PadFrame<pixel_t>(f0padded, env);
    CopyFrame<pixel_t>(f1, f1padded, env);
    PadFrame<pixel_t>(f1padded, env);

    AnalyzeFrame<pixel_t>(f0padded, f1padded, fflag, &prmY, &prmC, env);
    MergeUVFlags(fflag, env); // UV判定結果をYにマージ
    CountFlags(fflag, dst, parity, env);

    return dst;
  }

public:
  KFMFrameAnalyze(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
    , padvi(vi)
    , flagvi()
  {
    padvi.height += VPAD * 2;

    int out_bytes = sizeof(FMCount) * 2;
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 16;
    vi.height = nblocks(out_bytes, vi.width * 4);

    flagvi.pixel_type = Get8BitType(srcvi);
    flagvi.width = srcvi.width;
    flagvi.height = srcvi.height;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KFMFrameAnalyze] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyze(
      args[0].AsClip(),       // clip
      args[1].AsInt(15),      // threshMY
      args[2].AsInt(7),       // threshSY
      args[3].AsInt(20),      // threshMC
      args[4].AsInt(8),       // threshSC
      env
    );
  }
};

class KFMFrameAnalyzeCheck : public GenericVideoFilter
{
  PClip clipB;
public:
  KFMFrameAnalyzeCheck(PClip clipA, PClip clipB, IScriptEnvironment* env)
    : GenericVideoFilter(clipA)
    , clipB(clipB)
  {}

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    PVideoFrame frameA = child->GetFrame(n, env);
    PVideoFrame frameB = clipB->GetFrame(n, env);

    const FMCount* fmcntA = reinterpret_cast<const FMCount*>(frameA->GetReadPtr());
    const FMCount* fmcntB = reinterpret_cast<const FMCount*>(frameB->GetReadPtr());

    if (memcmp(fmcntA, fmcntB, sizeof(FMCount) * 2)) {
      env->ThrowError("[KFMFrameAnalyzeCheck] Unmatch !!!");
    }

    return frameA;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyzeCheck(
      args[0].AsClip(),       // clipA
      args[1].AsClip(),       // clipB
      env
    );
  }
};

class KFMFrameAnalyzeShow : public KFMFilterBase
{
  typedef uint8_t pixel_t;

  VideoInfo padvi;
  VideoInfo flagvi;

  FrameAnalyzeParam prmY;
  FrameAnalyzeParam prmC;

  int threshMY;
  int threshSY;
  int threshLSY;
  int threshMC;
  int threshSC;
  int threshLSC;

  int logUVx;
  int logUVy;

  void VisualizeFlags(PVideoFrame& dst, PVideoFrame& fflag, PNeoEnv env)
  {
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
    for (int y = 0; y < vi.height; ++y) {
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

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    PVideoFrame f0 = child->GetFrame(n, env);
    PVideoFrame f1 = child->GetFrame(n + 1, env);
    PVideoFrame f0padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame f1padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame fflag = env->NewVideoFrame(flagvi);
    PVideoFrame dst = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(f0, f0padded, env);
    PadFrame<pixel_t>(f0padded, env);
    CopyFrame<pixel_t>(f1, f1padded, env);
    PadFrame<pixel_t>(f1padded, env);

    AnalyzeFrame<pixel_t>(f0padded, f1padded, fflag, &prmY, &prmC, env);
    MergeUVFlags(fflag, env); // UV判定結果をYにマージ
    VisualizeFlags(dst, fflag, env);

    return dst;
  }

public:
  KFMFrameAnalyzeShow(PClip clip, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , padvi(vi)
    , flagvi()
  {
    padvi.height += VPAD * 2;

    flagvi.pixel_type = Get8BitType(srcvi);
    flagvi.width = srcvi.width;
    flagvi.height = srcvi.height;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KFMFrameDev] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMFrameAnalyzeShow(
      args[0].AsClip(),       // clip
      args[1].AsInt(10),       // threshMY
      args[2].AsInt(10),       // threshSY
      args[3].AsInt(10),       // threshMC
      args[4].AsInt(10),       // threshSC
      env
    );
  }
};

class KTelecine : public KFMFilterBase
{
  PClip fmclip;
  bool show;

  PulldownPatterns patterns;

  template <typename pixel_t>
  void CopyField(bool top, PVideoFrame* const * frames, const PVideoFrame& dst, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    PVideoFrame& frame0 = *frames[0];
    const vpixel_t* src0Y = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_Y));
    const vpixel_t* src0U = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_U));
    const vpixel_t* src0V = reinterpret_cast<const vpixel_t*>(frame0->GetReadPtr(PLANAR_V));
    vpixel_t* dstY = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_Y));
    vpixel_t* dstU = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_U));
    vpixel_t* dstV = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = frame0->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = frame0->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (!top) {
      src0Y += pitchY;
      src0U += pitchUV;
      src0V += pitchUV;
      dstY += pitchY;
      dstU += pitchUV;
      dstV += pitchUV;
    }

    if (frames[1] == nullptr) {
      if (IS_CUDA) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
        dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
        kl_copy << <blocks, threads >> >(dstY, src0Y, width4, vi.height / 2, pitchY * 2);
        DEBUG_SYNC;
        kl_copy << <blocksUV, threads >> >(dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
        kl_copy << <blocksUV, threads >> >(dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
      }
      else {
        cpu_copy(dstY, src0Y, width4, vi.height / 2, pitchY * 2);
        cpu_copy(dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
        cpu_copy(dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
      }
    }
    else {
      PVideoFrame& frame1 = *frames[1];
      const vpixel_t* src1Y = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_Y));
      const vpixel_t* src1U = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_U));
      const vpixel_t* src1V = reinterpret_cast<const vpixel_t*>(frame1->GetReadPtr(PLANAR_V));

      if (!top) {
        src1Y += pitchY;
        src1U += pitchUV;
        src1V += pitchUV;
      }

      if (IS_CUDA) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
        dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
        kl_average << <blocks, threads >> >(dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
        DEBUG_SYNC;
        kl_average << <blocksUV, threads >> >(dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
        kl_average << <blocksUV, threads >> >(dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
      }
      else {
        cpu_average(dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
        cpu_average(dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
        cpu_average(dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
      }
    }

  }

  template <typename pixel_t>
  PVideoFrame CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, int parity, PNeoEnv env)
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

      // 3フィールドのときは重複フィールドを平均化する

      PVideoFrame* srct[2] = { 0 };
      PVideoFrame* srcb[2] = { 0 };

      if (parity) {
        srct[0] = &nxt;
        srcb[0] = &cur;
        if (fnum >= 3) {
          if (fstart == 0) {
            srct[1] = &cur;
          }
          else {
            srcb[1] = &nxt;
          }
        }
      }
      else {
        srct[0] = &cur;
        srcb[0] = &nxt;
        if (fnum >= 3) {
          if (fstart == 0) {
            srcb[1] = &cur;
          }
          else {
            srct[1] = &nxt;
          }
        }
      }

      CopyField<pixel_t>(true, srct, dst, env);
      CopyField<pixel_t>(false, srcb, dst, env);

      return dst;
    }
  }

  void DrawInfo(PVideoFrame& dst, int pattern, float cost, int fnum, PNeoEnv env) {
    env->MakeWritable(&dst);

    char buf[100]; sprintf(buf, "KFM: %d (%.1f) - %d", pattern, cost, fnum);
    DrawText(dst, true, 0, 0, buf);
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    int cycleIndex = n / 4;
    int parity = child->GetParity(cycleIndex * 5);
    PVideoFrame fm = fmclip->GetFrame(cycleIndex, env);
    int pattern = (int)fm->GetProperty("KFM_Pattern")->GetInt();
    float cost = (float)fm->GetProperty("KFM_Cost")->GetFloat();
    Frame24Info frameInfo = patterns.GetFrame24(pattern, n);

    int fstart = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
    PVideoFrame out = CreateWeaveFrame<pixel_t>(child, 0, fstart, frameInfo.numFields, parity, env);

    if (sizeof(pixel_t) == 1 && !IS_CUDA && show) {
      // 8bit CPUにしか対応していない
      DrawInfo(out, pattern, cost, frameInfo.numFields, env);
    }

    return out;
  }

public:
  KTelecine(PClip child, PClip fmclip, bool show, IScriptEnvironment* env)
    : KFMFilterBase(child)
    , fmclip(fmclip)
    , show(show)
  {
    // フレームレート
    vi.MulDivFPS(4, 5);
    vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KTelecine] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KTelecine(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // fmclip
      args[2].AsBool(false),  // show
      env
    );
  }
};

enum {
  DC_OVERLAP = 4,
  DC_BLOCK_SIZE = 8,
  DC_BLOCK_TH_W = 8,
  DC_BLOCK_TH_H = 8,
};

template <typename pixel_t, typename vpixel_t>
void cpu_detect_combe(pixel_t* flagp, int fpitch,
  const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int shift)
{
  for (int by = 0; by < nBlkY - 1; ++by) {
    for (int bx = 0; bx < nBlkX - 1; ++bx) {
      int sum = 0;
      for (int tx = 0; tx < 2; ++tx) {
        int x = bx * DC_OVERLAP / 4 + tx;
        int y = by * DC_OVERLAP;
        auto L0 = srcp[x + (y + 0) * pitch];
        auto L1 = srcp[x + (y + 1) * pitch];
        auto L2 = srcp[x + (y + 2) * pitch];
        auto L3 = srcp[x + (y + 3) * pitch];
        auto L4 = srcp[x + (y + 4) * pitch];
        auto L5 = srcp[x + (y + 5) * pitch];
        auto L6 = srcp[x + (y + 6) * pitch];
        auto L7 = srcp[x + (y + 7) * pitch];
        int4 diff8 = absdiff(L0, L7);
        int4 diffT = absdiff(L0, L1) + absdiff(L1, L2) + absdiff(L2, L3) + absdiff(L3, L4) + absdiff(L4, L5) + absdiff(L5, L6) + absdiff(L6, L7) - diff8;
        int4 diffE = absdiff(L0, L2) + absdiff(L2, L4) + absdiff(L4, L6) + absdiff(L6, L7) - diff8;
        int4 diffO = absdiff(L0, L1) + absdiff(L1, L3) + absdiff(L3, L5) + absdiff(L5, L7) - diff8;
        int4 score = diffT - diffE - diffO;
        sum += score.x + score.y + score.z + score.w;
      }
      flagp[(bx + 1) + (by + 1) * fpitch] = clamp(sum >> shift, 0, 255);
    }
  }
}

template <typename pixel_t, typename vpixel_t>
__global__ void kl_detect_combe(pixel_t* flagp, int fpitch,
  const vpixel_t* srcp, int pitch, int nBlkX, int nBlkY, int shift)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x * DC_BLOCK_TH_W + threadIdx.y;
  int by = blockIdx.y * DC_BLOCK_TH_H + threadIdx.z;

  if (bx < nBlkX - 1 && by < nBlkY - 1) {
    int x = bx * DC_OVERLAP / 4 + tx;
    int y = by * DC_OVERLAP;
    auto L0 = srcp[x + (y + 0) * pitch];
    auto L1 = srcp[x + (y + 1) * pitch];
    auto L2 = srcp[x + (y + 2) * pitch];
    auto L3 = srcp[x + (y + 3) * pitch];
    auto L4 = srcp[x + (y + 4) * pitch];
    auto L5 = srcp[x + (y + 5) * pitch];
    auto L6 = srcp[x + (y + 6) * pitch];
    auto L7 = srcp[x + (y + 7) * pitch];
    int4 diff8 = absdiff(L0, L7);
    int4 diffT = absdiff(L0, L1) + absdiff(L1, L2) + absdiff(L2, L3) + absdiff(L3, L4) + absdiff(L4, L5) + absdiff(L5, L6) + absdiff(L6, L7) - diff8;
    int4 diffE = absdiff(L0, L2) + absdiff(L2, L4) + absdiff(L4, L6) + absdiff(L6, L7) - diff8;
    int4 diffO = absdiff(L0, L1) + absdiff(L1, L3) + absdiff(L3, L5) + absdiff(L5, L7) - diff8;
    int4 score = diffT - diffE - diffO;
    int sum = score.x + score.y + score.z + score.w;
#if CUDART_VERSION >= 9000
    sum += __shfl_down_sync(0xffffffff, sum, 1);
#else
    sum += __shfl_down(sum, 1);
#endif
    if (tx == 0) {
      flagp[(bx + 1) + (by + 1) * fpitch] = clamp(sum >> shift, 0, 255);
    }
  }
}

__device__ __host__ int BinomialMerge(int a, int b, int c, int d, int e, int thresh)
{
  int minv = min(a, min(b, min(c, min(d, e))));
  int maxv = max(a, max(b, max(c, max(d, e))));
  if (maxv - minv < thresh) {
    return (b + 2 * c + d + 2) >> 2;
  }
  return c;
}

template <typename pixel_t>
void cpu_remove_combe2(pixel_t* dst,
  const pixel_t* src, int width, int height, int pitch,
  const pixel_t* combe, int c_pitch, int thcombe, int thdiff)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int score = combe[(x >> 2) + (y >> 2) * c_pitch];
      if (score >= thcombe) {
        dst[x + y * pitch] = BinomialMerge(
          src[x + (y - 2) * pitch],
          src[x + (y - 1) * pitch],
          src[x + y * pitch],
          src[x + (y + 1) * pitch],
          src[x + (y + 2) * pitch],
          thdiff);
      }
      else {
        dst[x + y * pitch] = src[x + y * pitch];
      }
    }
  }
}

template <typename pixel_t>
__global__ void kl_remove_combe2(pixel_t* dst,
  const pixel_t* src, int width, int height, int pitch,
  const pixel_t* combe, int c_pitch, int thcombe, int thdiff)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int score = combe[(x >> 2) + (y >> 2) * c_pitch];
    if (score >= thcombe) {
      dst[x + y * pitch] = BinomialMerge(
        src[x + (y - 2) * pitch],
        src[x + (y - 1) * pitch],
        src[x + y * pitch],
        src[x + (y + 1) * pitch],
        src[x + (y + 2) * pitch],
        thdiff);
    }
    else {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

void cpu_max_extend_blocks(uint8_t* dstp, int pitch, int nBlkX, int nBlkY)
{
  for (int by = 1; by < nBlkY; ++by) {
    dstp[0 + by * pitch] = dstp[0 + 1 + (by + 0) * pitch];
    for (int bx = 1; bx < nBlkX - 1; ++bx) {
      dstp[bx + by * pitch] = max(
        dstp[bx + by * pitch], dstp[bx + 1 + (by + 0) * pitch]);
    }
  }
  for (int bx = 0; bx < nBlkX; ++bx) {
    dstp[bx] = dstp[bx + pitch];
  }
  for (int by = 1; by < nBlkY - 1; ++by) {
    for (int bx = 0; bx < nBlkX; ++bx) {
      dstp[bx + by * pitch] = max(
        dstp[bx + by * pitch], dstp[bx + 0 + (by + 1) * pitch]);
    }
  }
}

__global__ void kl_max_extend_blocks_h(uint8_t* dstp, const uint8_t* srcp, int pitch, int nBlkX, int nBlkY)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;

  if (bx < nBlkX && by < nBlkY) {
    if (bx == nBlkX - 1) {
      // 書き込む予定がないところにソースをコピーする
      dstp[bx + by * pitch] = srcp[bx + by * pitch];
    }
    else if (bx == 0) {
      dstp[bx + by * pitch] = srcp[bx + 1 + (by + 0) * pitch];
    }
    else {
      dstp[bx + by * pitch] = max(
        srcp[bx + 0 + (by + 0) * pitch], srcp[bx + 1 + (by + 0) * pitch]);
    }
  }
}

__global__ void kl_max_extend_blocks_v(uint8_t* dstp, const uint8_t* srcp, int pitch, int nBlkX, int nBlkY)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;

  if (bx < nBlkX && by < nBlkY) {
    if (by == nBlkY - 1) {
      // 書き込む予定がないところにソースをコピーする
      dstp[bx + by * pitch] = srcp[bx + by * pitch];
    }
    else if (by == 0) {
      dstp[bx + by * pitch] = srcp[bx + 0 + (by + 1) * pitch];
    }
    else {
      dstp[bx + by * pitch] = max(
        srcp[bx + 0 + (by + 0) * pitch], srcp[bx + 0 + (by + 1) * pitch]);
    }
  }
}

template <typename pixel_t>
void cpu_combe_to_flag(pixel_t* flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* combe, int cpitch)
{
  for (int y = 0; y < nBlkY; ++y) {
    for (int x = 0; x < nBlkX; ++x) {
      flag[x + y * fpitch] =
        (combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
          combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
          combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
          combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
    }
  }
}

template <typename pixel_t>
__global__ void kl_combe_to_flag(pixel_t* flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* combe, int cpitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nBlkX && y < nBlkY) {
    flag[x + y * fpitch] =
      (combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
        combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
        combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
        combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
  }
}

template <typename pixel_t>
void cpu_sum_box3x3(pixel_t* dst, pixel_t* src, int width, int height, int pitch, int maxv)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
        src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
        src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
      dst[x + y * pitch] = min(sumv >> 2, maxv); // 適当に1/4する
    }
  }
}

template <typename pixel_t>
__global__ void kl_sum_box3x3(pixel_t* dst, pixel_t* src, int width, int height, int pitch, int maxv)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
      src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
      src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
    dst[x + y * pitch] = min(sumv >> 2, maxv); // 適当に1/4する
  }
}

class KRemoveCombeCheck : public GenericVideoFilter
{
  PClip clipB;
  int nBlkX, nBlkY;
public:
  KRemoveCombeCheck(PClip clipA, PClip clipB, IScriptEnvironment* env)
    : GenericVideoFilter(clipA)
    , clipB(clipB)
  {
    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    PVideoFrame frameA = child->GetFrame(n, env)->GetProperty(COMBE_FLAG_STR)->GetFrame();
    PVideoFrame frameB = clipB->GetFrame(n, env)->GetProperty(COMBE_FLAG_STR)->GetFrame();

    const uint8_t* fmcntA = reinterpret_cast<const uint8_t*>(frameA->GetReadPtr());
    const uint8_t* fmcntB = reinterpret_cast<const uint8_t*>(frameB->GetReadPtr());

    for (int by = 0; by < nBlkY; ++by) {
      for (int bx = 0; bx < nBlkX; ++bx) {
        if (fmcntA[bx + by * nBlkX] != fmcntB[bx + by * nBlkX]) {
          env->ThrowError("[KRemoveCombeCheck] Unmatch !!!");
        }
      }
    }

    return frameA;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KRemoveCombeCheck(
      args[0].AsClip(),       // clipA
      args[1].AsClip(),       // clipB
      env
    );
  }
};

class KRemoveCombe : public KFMFilterBase
{
  VideoInfo padvi;
  VideoInfo combvi;
  VideoInfo blockvi;

  float thsmooth;
  float smooth;
  bool detect_uv;
  bool show;
  float thcombe;

  template <typename pixel_t>
  void DetectCombe(PVideoFrame& src, PVideoFrame& combe, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* srcY = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_Y));
    const vpixel_t* srcU = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_U));
    const vpixel_t* srcV = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(PLANAR_V));
    uint8_t* combeY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
    uint8_t* combeU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
    uint8_t* combeV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(vpixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(vpixel_t);
    int fpitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int fpitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
    int widthUV = combvi.width >> logUVx;
    int heightUV = combvi.height >> logUVy;

    int shift = vi.BitsPerComponent() - 8 + 4;

    if (IS_CUDA) {
      dim3 threads(2, DC_BLOCK_TH_W, DC_BLOCK_TH_H);
      dim3 blocks(nblocks(combvi.width, DC_BLOCK_TH_W), nblocks(combvi.height, DC_BLOCK_TH_H));
      dim3 blocksUV(nblocks(widthUV, DC_BLOCK_TH_W), nblocks(heightUV, DC_BLOCK_TH_H));
      kl_detect_combe << <blocks, threads >> >(combeY, fpitchY, srcY, pitchY, combvi.width, combvi.height, shift);
      DEBUG_SYNC;
      if (detect_uv) {
        kl_detect_combe << <blocksUV, threads >> >(combeU, fpitchUV, srcU, pitchUV, widthUV, heightUV, shift);
        DEBUG_SYNC;
        kl_detect_combe << <blocksUV, threads >> >(combeV, fpitchUV, srcV, pitchUV, widthUV, heightUV, shift);
        DEBUG_SYNC;
      }
    }
    else {
      cpu_detect_combe(combeY, fpitchY, srcY, pitchY, combvi.width, combvi.height, shift);
      if (detect_uv) {
        cpu_detect_combe(combeU, fpitchUV, srcU, pitchUV, widthUV, heightUV, shift);
        cpu_detect_combe(combeV, fpitchUV, srcV, pitchUV, widthUV, heightUV, shift);
      }
    }
  }

  void ExtendBlocks(PVideoFrame& dst, PVideoFrame& tmp, PNeoEnv env)
  {
    uint8_t* tmpY = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_Y));
    uint8_t* tmpU = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_U));
    uint8_t* tmpV = reinterpret_cast<uint8_t*>(tmp->GetWritePtr(PLANAR_V));
    uint8_t* dstY = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_Y));
    uint8_t* dstU = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_U));
    uint8_t* dstV = reinterpret_cast<uint8_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = tmp->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int pitchUV = tmp->GetPitch(PLANAR_U) / sizeof(uint8_t);
    int widthUV = combvi.width >> logUVx;
    int heightUV = combvi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(combvi.width, threads.x), nblocks(combvi.height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_max_extend_blocks_h << <blocks, threads >> >(tmpY, dstY, pitchY, combvi.width, combvi.height);
      kl_max_extend_blocks_v << <blocks, threads >> >(dstY, tmpY, pitchY, combvi.width, combvi.height);
      DEBUG_SYNC;
      if (detect_uv) {
        kl_max_extend_blocks_h << <blocksUV, threads >> > (tmpU, dstU, pitchUV, widthUV, heightUV);
        kl_max_extend_blocks_v << <blocksUV, threads >> > (dstU, tmpU, pitchUV, widthUV, heightUV);
        DEBUG_SYNC;
        kl_max_extend_blocks_h << <blocksUV, threads >> > (tmpV, dstV, pitchUV, widthUV, heightUV);
        kl_max_extend_blocks_v << <blocksUV, threads >> > (dstV, tmpV, pitchUV, widthUV, heightUV);
        DEBUG_SYNC;
      }
    }
    else {
      cpu_max_extend_blocks(dstY, pitchY, combvi.width, combvi.height);
      if (detect_uv) {
        cpu_max_extend_blocks(dstU, pitchUV, widthUV, heightUV);
        cpu_max_extend_blocks(dstV, pitchUV, widthUV, heightUV);
      }
    }
  }

  void MergeUVCoefs(PVideoFrame& combe, PNeoEnv env)
  {
    uint8_t* fY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
    uint8_t* fU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
    uint8_t* fV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));
    int pitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int pitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(combvi.width, threads.x), nblocks(combvi.height, threads.y));
      kl_merge_uvcoefs << <blocks, threads >> >(fY,
        fU, fV, combvi.width, combvi.height, pitchY, pitchUV, logUVx, logUVy);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_uvcoefs(fY,
        fU, fV, combvi.width, combvi.height, pitchY, pitchUV, logUVx, logUVy);
    }
  }

  void ApplyUVCoefs(PVideoFrame& combe, PNeoEnv env)
  {
    uint8_t* fY = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_Y));
    uint8_t* fU = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_U));
    uint8_t* fV = reinterpret_cast<uint8_t*>(combe->GetWritePtr(PLANAR_V));
    int pitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int pitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
    int widthUV = combvi.width >> logUVx;
    int heightUV = combvi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_apply_uvcoefs_420 << <blocks, threads >> >(fY,
        fU, fV, widthUV, heightUV, pitchY, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_apply_uvcoefs_420(fY, fU, fV, widthUV, heightUV, pitchY, pitchUV);
    }
  }

  template <typename pixel_t>
  void RemoveCombe(PVideoFrame& dst, PVideoFrame& src, PVideoFrame& combe, int thcombe, int thdiff, PNeoEnv env)
  {
    const uint8_t* combeY = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
    const uint8_t* combeU = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_U));
    const uint8_t* combeV = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_V));
    const pixel_t* srcY = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_Y));
    const pixel_t* srcU = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_U));
    const pixel_t* srcV = reinterpret_cast<const pixel_t*>(src->GetReadPtr(PLANAR_V));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int pitchY = src->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int pitchUV = src->GetPitch(PLANAR_U) / sizeof(pixel_t);
    int fpitchY = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int fpitchUV = combe->GetPitch(PLANAR_U) / sizeof(uint8_t);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_remove_combe2 << <blocks, threads >> >(dstY, srcY, vi.width, vi.height, pitchY, combeY, fpitchY, thcombe, thdiff);
      DEBUG_SYNC;
      kl_remove_combe2 << <blocksUV, threads >> >(dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thcombe, thdiff);
      DEBUG_SYNC;
      kl_remove_combe2 << <blocksUV, threads >> >(dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thcombe, thdiff);
      DEBUG_SYNC;
    }
    else {
      cpu_remove_combe2(dstY, srcY, vi.width, vi.height, pitchY, combeY, fpitchY, thcombe, thdiff);
      cpu_remove_combe2(dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thcombe, thdiff);
      cpu_remove_combe2(dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thcombe, thdiff);
    }
  }

  template <typename pixel_t>
  void VisualizeCombe(PVideoFrame& dst, PVideoFrame& combe, int thresh, PNeoEnv env)
  {
    // 判定結果を表示
    int blue[] = { 73, 230, 111 };

    const uint8_t* combep = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));

    int combePitch = combe->GetPitch(PLANAR_Y) / sizeof(uint8_t);
    int dstPitchY = dst->GetPitch(PLANAR_Y) / sizeof(pixel_t);
    int dstPitchUV = dst->GetPitch(PLANAR_U) / sizeof(pixel_t);

    // 色を付ける
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int score = combep[(x >> 2) + (y >> 2) * combePitch];

        int* color = nullptr;
        if (score >= thresh) {
          color = blue;
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

  void MakeSwitchFlag(PVideoFrame& flag, PVideoFrame& flagtmp, PVideoFrame& combe, PNeoEnv env)
  {
    const uint8_t* srcp = reinterpret_cast<const uint8_t*>(combe->GetReadPtr(PLANAR_Y));
    uint8_t* flagp = reinterpret_cast<uint8_t*>(flag->GetWritePtr());
    uint8_t* flagtmpp = reinterpret_cast<uint8_t*>(flagtmp->GetWritePtr());

    int height = flag->GetHeight();
    int width = flag->GetRowSize();
    int fpitch = flag->GetPitch();
    int cpitch = combe->GetPitch();

    if (IS_CUDA) {
      dim3 threads(16, 8);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      kl_combe_to_flag << <blocks, threads >> >(
        flagp, width, height, fpitch, srcp, cpitch);
      DEBUG_SYNC;
      kl_sum_box3x3 << <blocks, threads >> >(
        flagtmpp, flagp, width, height, fpitch, 255);
      DEBUG_SYNC;
      kl_sum_box3x3 << <blocks, threads >> >(
        flagp, flagtmpp, width, height, fpitch, 255);
      DEBUG_SYNC;
    }
    else {
      cpu_combe_to_flag(flagp, width, height, fpitch, srcp, cpitch);
      cpu_sum_box3x3(flagtmpp, flagp, width, height, fpitch, 255);
      cpu_sum_box3x3(flagp, flagtmpp, width, height, fpitch, 255);
    }
  }

  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    typedef uint8_t pixel_t;

    int frame_align = (int)env->GetProperty(AEP_FRAME_ALIGN);

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame padded = OffsetPadFrame(env->NewVideoFrame(padvi), env);
    PVideoFrame dst = env->NewVideoFrame(vi);
    PVideoFrame combe = env->NewVideoFrame(combvi);
    PVideoFrame combetmp = env->NewVideoFrame(combvi);
    PVideoFrame flag = NewSwitchFlagFrame(vi, frame_align, 2, env);
    PVideoFrame flagtmp = NewSwitchFlagFrame(vi, frame_align, 2, env);

    CopyFrame<pixel_t>(src, padded, env);
    PadFrame<pixel_t>(padded, env);
    DetectCombe<pixel_t>(padded, combe, env);
    ExtendBlocks(combe, combetmp, env);
    if (detect_uv) {
      MergeUVCoefs(combe, env);
    }
    ApplyUVCoefs(combe, env);
    RemoveCombe<pixel_t>(dst, padded, combe, (int)thsmooth, (int)smooth, env);
    DetectCombe<pixel_t>(dst, combe, env);
    ExtendBlocks(combe, combetmp, env);
    if (detect_uv) {
      MergeUVCoefs(combe, env);
    }
    MakeSwitchFlag(flag, flagtmp, combe, env);
    dst->SetProperty(COMBE_FLAG_STR, flag);

    if (!IS_CUDA && show) {
      VisualizeCombe<pixel_t>(dst, combe, (int)thcombe, env);
      return dst;
    }

    return dst;
  }

public:
  KRemoveCombe(PClip clip, float thsmooth, float smooth, bool uv, bool show, float thcombe, IScriptEnvironment* env)
    : KFMFilterBase(clip)
    , padvi(vi)
    , blockvi(vi)
    , thsmooth(thsmooth)
    , smooth(smooth)
    , detect_uv(uv)
    , show(show)
    , thcombe(thcombe)
  {
    if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

    padvi.height += VPAD * 2;

    combvi.width = vi.width / DC_OVERLAP;
    combvi.height = vi.height / DC_OVERLAP;
    combvi.pixel_type = Get8BitType(vi);

    blockvi.width = nblocks(vi.width, OVERLAP);
    blockvi.height = nblocks(vi.height, OVERLAP);
    blockvi.pixel_type = VideoInfo::CS_Y8;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT(n, env);
    case 2:
      return GetFrameT(n, env);
    default:
      env->ThrowError("[KRemoveCombe] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KRemoveCombe(
      args[0].AsClip(),       // source
      (float)args[1].AsFloat(30), // thsmooth
      (float)args[2].AsFloat(100), // smooth
      args[3].AsBool(false), // uv
      args[4].AsBool(false), // show
      (float)args[5].AsFloat(100), // thcombe
      env
    );
  }
};

void AddFuncCombingAnalyze(IScriptEnvironment* env)
{
  env->AddFunction("KFMFrameAnalyzeShow", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyzeShow::Create, 0);
  env->AddFunction("KFMFrameAnalyze", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMFrameAnalyze::Create, 0);

  env->AddFunction("KFMFrameAnalyzeCheck", "cc", KFMFrameAnalyzeCheck::Create, 0);

  env->AddFunction("KTelecine", "cc[show]b", KTelecine::Create, 0);
  env->AddFunction("KRemoveCombe", "c[thsmooth]f[smooth]f[uv]b[show]b[thcombe]f", KRemoveCombe::Create, 0);
  env->AddFunction("KRemoveCombeCheck", "cc", KRemoveCombeCheck::Create, 0);
}
