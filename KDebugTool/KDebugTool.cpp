#define _CRT_SECURE_NO_WARNINGS
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

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
  const bool alpha;
  const double thresh;

  int offX;
  int offY;

  void PrintMissMatch(int a, int b, int x, int y) {
    printf("miss match %d vs %d at (%d,%d)\n", a, b, x, y);
  }
  void PrintMissMatch(float a, float b, int x, int y) {
    printf("miss match %f vs %f at (%d,%d)\n", a, b, x, y);
  }

  template <typename pixel_t>
  void ComparePlane(
    pixel_t* d, int dpitch, 
    const pixel_t* a, int apitch,
    const pixel_t* b, int bpitch, 
    int width, int height, PNeoEnv env)
  {
    pixel_t thresh = (pixel_t)this->thresh;
		if (sizeof(pixel_t) == 1) {
			//DebugWriteBitmap("bob-ref-%d.bmp", (const uint8_t*)a, width, height, apitch, 1);
			//DebugWriteBitmap("bob-cuda-%d.bmp", (const uint8_t*)b, width, height, bpitch, 1);
		}

    for (int y = offY; y < height; ++y) {
      for (int x = offX; x < width; ++x) {
        pixel_t diff = std::abs(a[x + y * apitch] - b[x + y * bpitch]);
        if (diff > thresh) {
          PrintMissMatch(a[x + y * apitch], b[x + y * bpitch], x, y);
          env->ThrowError("[ImageCompare] âÊëúÇ™àÍívÇµÇ‹ÇπÇÒÅBÉeÉXÉgé∏îs");
        }
        d[x + y * dpitch] = diff;
      }
    }
  }

  template <typename pixel_t>
  void ComparePlanar(PVideoFrame& dst, PVideoFrame& frame1, PVideoFrame& frame2, PNeoEnv env)
  {
    int isRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    static const int planesYUV[] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    static const int planesRGB[] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int *planes = isRGB ? planesRGB : planesYUV;

    pixel_t* dstY = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[0]));
    pixel_t* dstU = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[1]));
    pixel_t* dstV = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[2]));
    pixel_t* dstA = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[3]));
    const pixel_t* f1Y = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(planes[0]));
    const pixel_t* f1U = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(planes[1]));
    const pixel_t* f1V = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(planes[2]));
    const pixel_t* f1A = reinterpret_cast<const pixel_t*>(frame1->GetReadPtr(planes[3]));
    const pixel_t* f2Y = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(planes[0]));
    const pixel_t* f2U = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(planes[1]));
    const pixel_t* f2V = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(planes[2]));
    const pixel_t* f2A = reinterpret_cast<const pixel_t*>(frame2->GetReadPtr(planes[3]));

    ComparePlane<pixel_t>(
      dstY, dst->GetPitch(planes[0]) / sizeof(pixel_t),
      f1Y, frame1->GetPitch(planes[0]) / sizeof(pixel_t),
      f2Y, frame2->GetPitch(planes[0]) / sizeof(pixel_t),
      dst->GetRowSize(planes[0]) / sizeof(pixel_t), dst->GetHeight(planes[0]), env);

    if (chroma) {
      if (frame1->GetPitch(planes[1])) {
        ComparePlane<pixel_t>(
          dstU, dst->GetPitch(planes[1]) / sizeof(pixel_t),
          f1U, frame1->GetPitch(planes[1]) / sizeof(pixel_t),
          f2U, frame2->GetPitch(planes[1]) / sizeof(pixel_t),
          dst->GetRowSize(planes[1]) / sizeof(pixel_t), dst->GetHeight(planes[1]), env);
      }
      if (frame1->GetPitch(planes[2])) {
        ComparePlane<pixel_t>(
          dstV, dst->GetPitch(planes[2]) / sizeof(pixel_t),
          f1V, frame1->GetPitch(planes[2]) / sizeof(pixel_t),
          f2V, frame2->GetPitch(planes[2]) / sizeof(pixel_t),
          dst->GetRowSize(planes[2]) / sizeof(pixel_t), dst->GetHeight(planes[2]), env);
      }
    }
    if (alpha && frame1->GetPitch(planes[2])) {
      ComparePlane<pixel_t>(
        dstY, dst->GetPitch(planes[3]) / sizeof(pixel_t),
        f1Y, frame1->GetPitch(planes[3]) / sizeof(pixel_t),
        f2Y, frame2->GetPitch(planes[3]) / sizeof(pixel_t),
        dst->GetRowSize(planes[3]) / sizeof(pixel_t), dst->GetHeight(planes[3]), env);
    }
  }

  template <typename pixel_t>
  void CompareRGB(PVideoFrame& dst, PVideoFrame& frameA, PVideoFrame& frameB, PNeoEnv env)
  {
    pixel_t* d = reinterpret_cast<pixel_t*>(dst->GetWritePtr());
    const pixel_t* a = reinterpret_cast<const pixel_t*>(frameA->GetReadPtr());
    const pixel_t* b = reinterpret_cast<const pixel_t*>(frameB->GetReadPtr());

    int dpitch = dst->GetPitch() / sizeof(pixel_t);
    int apitch = frameA->GetPitch() / sizeof(pixel_t);
    int bpitch = frameB->GetPitch() / sizeof(pixel_t);

    int width = vi.width;
    int height = vi.height;
    int el = ((vi.IsRGB24() || vi.IsRGB48()) ? 3 : 4);
    pixel_t thresh = (pixel_t)this->thresh;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width * el; ++x) {
        pixel_t diff = std::abs(a[x + y * apitch] - b[x + y * bpitch]);
        if (diff > thresh) {
          const char* color = "BGRA";
          printf("miss match %d vs %d at %c(%d,%d)\n", a[x + y * apitch], b[x + y * bpitch],
            color[x % el], x / el , y);
          env->ThrowError("[ImageCompare] âÊëúÇ™àÍívÇµÇ‹ÇπÇÒÅBÉeÉXÉgé∏îs");
        }
        d[x + y * dpitch] = diff;
      }
    }
  }

public:
  ImageCompare(PClip child1, PClip child2, double thresh, bool chroma, bool alpha, int offX, int offY, PNeoEnv env)
    : GenericVideoFilter(child1)
    , child2(child2)
    , thresh((vi.ComponentSize() == 4) ? thresh / 65536 : thresh)
    , chroma(chroma)
    , alpha(alpha)
    , offX(offX)
    , offY(offY)
  {
    if (vi.IsYUY2()) {
      env->ThrowError("[ImageCompare] YUY2 is not supported");
    }

    VideoInfo vi2 = child2->GetVideoInfo();

    if (vi.width != vi2.width) {
      env->ThrowError("[ImageCompare] different width");
    }
    if (vi.height != vi2.height) {
      env->ThrowError("[ImageCompare] different height");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    PVideoFrame frame1 = child->GetFrame(n, env);
    PVideoFrame frame2 = child2->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int nPixelSize = vi.ComponentSize();
    if (vi.IsPlanar()) {
      if (nPixelSize == 1) {
        ComparePlanar<uint8_t>(dst, frame1, frame2, env);
      }
      else if (nPixelSize == 2) {
        ComparePlanar<uint16_t>(dst, frame1, frame2, env);
      }
      else {
        ComparePlanar<float>(dst, frame1, frame2, env);
      }
    }
    else {
      if (nPixelSize == 1) {
        CompareRGB<uint8_t>(dst, frame1, frame2, env);
      }
      else if (nPixelSize == 2) {
        CompareRGB<uint16_t>(dst, frame1, frame2, env);
      }
    }

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new ImageCompare(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsFloat(2), // thresh
      args[3].AsBool(true), // chroma
      args[4].AsBool(true), // alpha
      args[5].AsInt(0), // offX
      args[6].AsInt(0), // offY
      env);
  }
};

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
  AVS_linkage = vectors;
  //init_console();
  
  env->AddFunction("Time", "c[name]s", Create_Time, 0);
  env->AddFunction("ImageCompare", "cc[thresh]f[chroma]b[alpha]b[offX]i[offY]i", ImageCompare::Create, 0);

  return "K Debug Plugin";
}
