#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <windows.h>
#include "avisynth.h"

#include <string>

#include "DebugWriter.h"

static void init_console()
{
  AllocConsole();
  freopen("CONOUT$", "w", stdout);
  freopen("CONIN$", "r", stdin);
}

class Time : public GenericVideoFilter {
  std::string name;
public:
  Time(PClip _child, const char* name, IScriptEnvironment* env)
    : GenericVideoFilter(_child)
    , name(name)
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    LARGE_INTEGER liBefore, liAfter, liFreq;

    QueryPerformanceCounter(&liBefore);

    PVideoFrame frame = child->GetFrame(n, env);

    QueryPerformanceCounter(&liAfter);
    QueryPerformanceFrequency(&liFreq);

    double sec = (double)(liAfter.QuadPart - liBefore.QuadPart) / liFreq.QuadPart;
    printf("[%5d] N:%5d %s: %.1f ms\n", GetCurrentThreadId(), n, name.c_str(), sec * 1000);

    return frame;
  }
};

AVSValue __cdecl Create_Time(AVSValue args, void* user_data, IScriptEnvironment* env) {
  return new Time(args[0].AsClip(), args[1].AsString("Time"), env);
}

class ImageCompare : GenericVideoFilter
{
  PClip child2;

  const bool chroma;
  const int thresh;

  template <typename pixel_t>
  void ComparePlanar(
    pixel_t* d, int dpitch, 
    const pixel_t* a, int apitch,
    const pixel_t* b, int bpitch, 
    int width, int height, IScriptEnvironment2* env)
  {
    //DebugWriteBitmap("bob-ref-%d.bmp", (const uint8_t*)a, width, height, pitch, 1);
    //DebugWriteBitmap("bob-cuda-%d.bmp", (const uint8_t*)b, width, height, pitch, 1);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        pixel_t diff = std::abs(a[x + y * apitch] - b[x + y * bpitch]);
        if (diff > thresh) {
          printf("miss match %d vs %d at (%d,%d)\n", a[x + y * apitch], b[x + y * bpitch], x, y);
          env->ThrowError("[ImageCompare] âÊëúÇ™àÍívÇµÇ‹ÇπÇÒÅBÉeÉXÉgé∏îs");
        }
        d[x + y * dpitch] = diff;
      }
    }
  }

  template <typename pixel_t>
  void CompareFrame(PVideoFrame& dst, PVideoFrame& frame1, PVideoFrame& frame2, IScriptEnvironment2* env)
  {
    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_Y));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_U));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(PLANAR_V));
    const pixel_t* f1Y = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(PLANAR_Y));
    const pixel_t* f1U = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(PLANAR_U));
    const pixel_t* f1V = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(PLANAR_V));
    const pixel_t* f2Y = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(PLANAR_Y));
    const pixel_t* f2U = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(PLANAR_U));
    const pixel_t* f2V = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(PLANAR_V));

    int nPixelShift = (sizeof(pixel_t) == 1) ? 0 : 1;
    int nUVShiftX = vi.GetPlaneWidthSubsampling(PLANAR_U);
    int nUVShiftY = vi.GetPlaneHeightSubsampling(PLANAR_U);

    ComparePlanar<pixel_t>(
      dstY, dst->GetPitch(PLANAR_Y) >> nPixelShift, 
      f1Y, frame1->GetPitch(PLANAR_Y) >> nPixelShift,
      f2Y, frame2->GetPitch(PLANAR_Y) >> nPixelShift,
      vi.width, vi.height, env);

    if (chroma) {
      ComparePlanar<pixel_t>(
        dstU, dst->GetPitch(PLANAR_U) >> nPixelShift,
        f1U, frame1->GetPitch(PLANAR_U) >> nPixelShift,
        f2U, frame2->GetPitch(PLANAR_U) >> nPixelShift,
        vi.width >> nUVShiftX, vi.height >> nUVShiftY, env);
      ComparePlanar<pixel_t>(
        dstV, dst->GetPitch(PLANAR_V) >> nPixelShift,
        f1V, frame1->GetPitch(PLANAR_V) >> nPixelShift,
        f2V, frame2->GetPitch(PLANAR_V) >> nPixelShift,
        vi.width >> nUVShiftX, vi.height >> nUVShiftY, env);
    }
  }

public:
  ImageCompare(PClip child1, PClip child2, int thresh, bool chroma, IScriptEnvironment2* env)
    : GenericVideoFilter(child1)
    , child2(child2)
    , thresh(thresh)
    , chroma(chroma)
  {
    //
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);

    PVideoFrame frame1 = child->GetFrame(n, env);
    PVideoFrame frame2 = child2->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int nPixelSize = vi.ComponentSize();
    if (nPixelSize == 1) {
      CompareFrame<uint8_t>(dst, frame1, frame2, env);
    }
    else {
      CompareFrame<uint16_t>(dst, frame1, frame2, env);
    }

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
    return new ImageCompare(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsInt(2), // thresh
      args[3].AsBool(true), // thresh
      env);
  }
};

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  //init_console();
  
  env->AddFunction("Time", "c[name]s", Create_Time, 0);
  env->AddFunction("ImageCompare", "cc[thresh]i[chroma]b", ImageCompare::Create, 0);

  return "K Debug Plugin";
}
