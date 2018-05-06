
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <memory>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"


template <typename vpixel_t>
void cpu_compare_frames(vpixel_t* dst,
  const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2, const vpixel_t* src3, const vpixel_t* src4,
  int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 a = to_int(src0[x + y * pitch]);
      int4 b = to_int(src1[x + y * pitch]);
      int4 c = to_int(src2[x + y * pitch]);
      int4 d = to_int(src3[x + y * pitch]);
      int4 e = to_int(src4[x + y * pitch]);

      int4 minv = min(min(a, b), min(c, min(d, e)));
      int4 maxv = max(max(a, b), max(c, max(d, e)));

      // ƒtƒ‰ƒOŠi”[
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_compare_frames(vpixel_t* dst,
  const vpixel_t* __restrict__ src0,
  const vpixel_t* __restrict__ src1,
  const vpixel_t* __restrict__ src2,
  const vpixel_t* __restrict__ src3,
  const vpixel_t* __restrict__ src4,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 a = to_int(src0[x + y * pitch]);
    int4 b = to_int(src1[x + y * pitch]);
    int4 c = to_int(src2[x + y * pitch]);
    int4 d = to_int(src3[x + y * pitch]);
    int4 e = to_int(src4[x + y * pitch]);

    int4 minv = min(min(a, b), min(c, min(d, e)));
    int4 maxv = max(max(a, b), max(c, max(d, e)));

    // ƒtƒ‰ƒOŠi”[
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(maxv - minv);
  }
}

class KTemporalDiff : public KFMFilterBase
{
  enum {
    DIST = 2,
    N_REFS = DIST * 2 + 1,
  };

  Frame GetRefFrame(int ref, PNeoEnv env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return child->GetFrame(ref, env);
  }

  template <typename pixel_t>
  void CompareFrames(Frame* frames, Frame& flag, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    const vpixel_t* srcY[N_REFS];
    const vpixel_t* srcU[N_REFS];
    const vpixel_t* srcV[N_REFS];

    for (int i = 0; i < N_REFS; ++i) {
      srcY[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_Y);
      srcU[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_U);
      srcV[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_V);
    }

    vpixel_t* dstY = flag.GetWritePtr<vpixel_t>(PLANAR_Y);
    vpixel_t* dstU = flag.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = flag.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchY = frames[0].GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = frames[0].GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_compare_frames << <blocks, threads >> >(dstY,
        srcY[0], srcY[1], srcY[2], srcY[3], srcY[4], width4, vi.height, pitchY);
      DEBUG_SYNC;
      kl_compare_frames << <blocksUV, threads >> >(dstU,
        srcU[0], srcU[1], srcU[2], srcU[3], srcU[4], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_compare_frames << <blocksUV, threads >> >(dstV,
        srcV[0], srcV[1], srcV[2], srcV[3], srcV[4], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_compare_frames(dstY, srcY[0], srcY[1], srcY[2], srcY[3], srcY[4], width4, vi.height, pitchY);
      cpu_compare_frames(dstU, srcU[0], srcU[1], srcU[2], srcU[3], srcU[4], width4UV, heightUV, pitchUV);
      cpu_compare_frames(dstV, srcV[0], srcV[1], srcV[2], srcV[3], srcV[4], width4UV, heightUV, pitchUV);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame frames[N_REFS];
    for (int i = 0; i < N_REFS; ++i) {
      frames[i] = GetRefFrame(i + n - DIST, env);
    }
    Frame diff = env->NewVideoFrame(vi);

    CompareFrames<pixel_t>(frames, diff, env);

    return diff.frame;
  }

public:
  KTemporalDiff(PClip clip30, PNeoEnv env)
    : KFMFilterBase(clip30)
  { }

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
      env->ThrowError("[KTemporalDiff] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KTemporalDiff(
      args[0].AsClip(),       // clip30
      env);
  }
};


template <typename vpixel_t>
void cpu_min_frames(vpixel_t* dst,
  const vpixel_t* src0, const vpixel_t* src1, const vpixel_t* src2,
  int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 a = to_int(src0[x + y * pitch]);
      int4 b = to_int(src1[x + y * pitch]);
      int4 c = to_int(src2[x + y * pitch]);

      int4 minv = min(a, min(b, c));

      // ƒtƒ‰ƒOŠi”[
      dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(minv);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_min_frames(vpixel_t* dst,
  const vpixel_t* __restrict__ src0,
  const vpixel_t* __restrict__ src1,
  const vpixel_t* __restrict__ src2,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 a = to_int(src0[x + y * pitch]);
    int4 b = to_int(src1[x + y * pitch]);
    int4 c = to_int(src2[x + y * pitch]);

    int4 minv = min(a, min(b, c));

    // ƒtƒ‰ƒOŠi”[
    dst[x + y * pitch] = VHelper<vpixel_t>::cast_to(minv);
  }
}

template <typename vpixel_t>
__global__ void kl_and_coefs(vpixel_t* dstp, const vpixel_t* __restrict__ diffp,
  int width, int height, int pitch, float invcombe, float invdiff)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    float4 combe = clamp(to_float(dstp[x + y * pitch]) * invcombe + (-1.0f), -0.5f, 0.5f);
    float4 diff = clamp(to_float(diffp[x + y * pitch]) * (-invdiff) + 1.0f, -0.5f, 0.5f);
    float4 tmp = max(combe + diff, 0.0f) * 128.0f + 0.5f;
    dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}


class KAnalyzeStatic : public KFMFilterBase
{
  enum {
    DIST = 1,
    N_DIFFS = DIST * 2 + 1,
  };

  PClip diffclip;
  PClip superclip;

  VideoInfo padvi;

  float thcombe;
  float thdiff;

  Frame GetDiffFrame(int ref, PNeoEnv env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return diffclip->GetFrame(ref, env);
  }

  template <typename pixel_t>
  void GetTemporalDiff(Frame* frames, Frame& flag, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    const vpixel_t* srcY[N_DIFFS];
    const vpixel_t* srcU[N_DIFFS];
    const vpixel_t* srcV[N_DIFFS];

    for (int i = 0; i < N_DIFFS; ++i) {
      srcY[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_Y);
      srcU[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_U);
      srcV[i] = frames[i].GetReadPtr<vpixel_t>(PLANAR_V);
    }

    vpixel_t* dstY = flag.GetWritePtr<vpixel_t>(PLANAR_Y);
    vpixel_t* dstU = flag.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = flag.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchY = frames[0].GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = frames[0].GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_min_frames << <blocks, threads >> >(dstY,
        srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
      DEBUG_SYNC;
      kl_min_frames << <blocksUV, threads >> >(dstU,
        srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_min_frames << <blocksUV, threads >> >(dstV,
        srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_min_frames(dstY, srcY[0], srcY[1], srcY[2], width4, vi.height, pitchY);
      cpu_min_frames(dstU, srcU[0], srcU[1], srcU[2], width4UV, heightUV, pitchUV);
      cpu_min_frames(dstV, srcV[0], srcV[1], srcV[2], width4UV, heightUV, pitchUV);
    }
  }

  template <typename pixel_t>
  void AndCoefs(Frame& dst, Frame& flagd, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* diffp = flagd.GetReadPtr<vpixel_t>();
    vpixel_t* dstp = dst.GetWritePtr<vpixel_t>();
    int pitch = dst.GetPitch<vpixel_t>();

    // dst: combe‚ ‚èƒtƒ‰ƒO
    // flagd: diff‚ ‚èƒtƒ‰ƒO
    float invcombe = 1.0f / thcombe;
    float invdiff = 1.0f / thdiff;
    int width4 = vi.width >> 2;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(vi.width, threads.x), nblocks(vi.height, threads.y));
      kl_and_coefs << <blocks, threads >> >(
        dstp, diffp, width4, vi.height, pitch, invcombe, invdiff);
      DEBUG_SYNC;
    }
    else {
      cpu_and_coefs(dstp, diffp, width4, vi.height, pitch, invcombe, invdiff);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame diffframes[N_DIFFS];
    for (int i = 0; i < N_DIFFS; ++i) {
      diffframes[i] = GetDiffFrame(i + n - DIST, env);
    }

    Frame padded;
    if (superclip) {
      padded = Frame(superclip->GetFrame(n, env), VPAD);
    }
    else {
      Frame f0 = child->GetFrame(n, env);
      padded = Frame(env->NewVideoFrame(padvi), VPAD);
      CopyFrame<pixel_t>(f0, padded, env);
      PadFrame<pixel_t>(padded, env);
    }

    Frame flagtmp = env->NewVideoFrame(vi);
    Frame flagc = env->NewVideoFrame(vi);
    Frame flagd = env->NewVideoFrame(vi);

    CompareFields<pixel_t>(padded, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagc, env);

    GetTemporalDiff<pixel_t>(diffframes, flagtmp, env);
    MergeUVCoefs<pixel_t>(flagtmp, env);
    ExtendCoefs<pixel_t>(flagtmp, flagd, env);

    AndCoefs<pixel_t>(flagc, flagd, env); // combe‚ ‚èdiff‚È‚µ -> flagc
    ApplyUVCoefs<pixel_t>(flagc, env);

    return flagc.frame;
  }

public:
  KAnalyzeStatic(PClip clip30, PClip diffclip, float thcombe, float thdiff, PClip pad, PNeoEnv env)
    : KFMFilterBase(clip30)
    , diffclip(diffclip)
    , thcombe(thcombe)
    , thdiff(thdiff)
    , padvi(vi)
    , superclip(pad)
  {
    if (logUVx != 1 || logUVy != 1) env->ThrowError("[KAnalyzeStatic] Unsupported format (only supports YV12)");

    padvi.height += VPAD * 2;
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
      env->ThrowError("[KAnalyzeStatic] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PClip clip30 = args[0].AsClip();
    PClip diffclip = env_->Invoke("KTemporalDiff", clip30).AsClip();
    return new KAnalyzeStatic(
      clip30,       // clip30
      diffclip,     // 
      (float)args[1].AsFloat(30),     // thcombe
      (float)args[2].AsFloat(15),     // thdiff
      args[3].AsClip(),        // pad
      env);
  }
};

template <typename vpixel_t>
void cpu_merge_static(
  vpixel_t* dstp, const vpixel_t* src60, const vpixel_t* src30, int pitch,
  const vpixel_t* flagp, int width, int height)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int4 coef = to_int(flagp[x + y * pitch]);
      int4 v30 = to_int(src30[x + y * pitch]);
      int4 v60 = to_int(src60[x + y * pitch]);
      int4 tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
      dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
    }
  }
}

template <typename vpixel_t>
__global__ void kl_merge_static(
  vpixel_t* dstp, const vpixel_t* src60, const vpixel_t* src30, int pitch,
  const vpixel_t* flagp, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int4 coef = to_int(flagp[x + y * pitch]);
    int4 v30 = to_int(src30[x + y * pitch]);
    int4 v60 = to_int(src60[x + y * pitch]);
    int4 tmp = (coef * v30 + (128 - coef) * v60 + 64) >> 7;
    dstp[x + y * pitch] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KMergeStatic : public KFMFilterBase
{
  PClip clip30;
  PClip sttclip;

  template <typename pixel_t>
  void MergeStatic(Frame& src60, Frame& src30, Frame& flag, Frame& dst, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    const vpixel_t* src60Y = src60.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src60U = src60.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src60V = src60.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* src30Y = src30.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src30U = src30.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src30V = src30.GetReadPtr<vpixel_t>(PLANAR_V);
    const vpixel_t* flagY = flag.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* flagU = flag.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* flagV = flag.GetReadPtr<vpixel_t>(PLANAR_V);
    vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
    vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchY = src60.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = src60.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(vi.height, threads.y));
      dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV, threads.y));
      kl_merge_static << <blocks, threads >> >(
        dstY, src60Y, src30Y, pitchY, flagY, width4, vi.height);
      DEBUG_SYNC;
      kl_merge_static << <blocksUV, threads >> >(
        dstU, src60U, src30U, pitchUV, flagU, width4UV, heightUV);
      DEBUG_SYNC;
      kl_merge_static << <blocksUV, threads >> >(
        dstV, src60V, src30V, pitchUV, flagV, width4UV, heightUV);
      DEBUG_SYNC;
    }
    else {
      cpu_merge_static(dstY, src60Y, src30Y, pitchY, flagY, width4, vi.height);
      cpu_merge_static(dstU, src60U, src30U, pitchUV, flagU, width4UV, heightUV);
      cpu_merge_static(dstV, src60V, src30V, pitchUV, flagV, width4UV, heightUV);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    int n30 = n >> 1;
    Frame flag = sttclip->GetFrame(n30, env);
    Frame frame60 = child->GetFrame(n, env);
    Frame frame30 = clip30->GetFrame(n30, env);
    Frame dst = env->NewVideoFrame(vi);

    CopyFrame<pixel_t>(frame60, dst, env);
    MergeStatic<pixel_t>(frame60, frame30, flag, dst, env);

    return dst.frame;
  }

public:
  KMergeStatic(PClip clip60, PClip clip30, PClip sttclip, PNeoEnv env)
    : KFMFilterBase(clip60)
    , clip30(clip30)
    , sttclip(sttclip)
  {
    VideoInfo srcvi = clip30->GetVideoInfo();
    if (vi.num_frames != srcvi.num_frames * 2) {
      env->ThrowError("[KMergeStatic] Num frames don't match");
    }
    if (!(GetDeviceTypes(clip60) & GetDeviceTypes(clip30))) {
      env->ThrowError("[KMergeStatic] Device unmatch. Two source must be same device.");
    }
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
      env->ThrowError("[KMergeStatic] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KMergeStatic(
      args[0].AsClip(),       // clip60
      args[1].AsClip(),       // clip30
      args[2].AsClip(),       // sttclip
      env);
  }
};

void AddFuncMergeStatic(IScriptEnvironment* env)
{
  env->AddFunction("KTemporalDiff", "c", KTemporalDiff::Create, 0);

  env->AddFunction("KAnalyzeStatic", "c[thcombe]f[thdiff]f[pad]c", KAnalyzeStatic::Create, 0);
  env->AddFunction("KMergeStatic", "ccc", KMergeStatic::Create, 0);
}
