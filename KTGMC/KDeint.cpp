#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#undef min
#undef max

#include "CommonFunctions.h"
#include "DeviceLocalData.h"
#include "DebugWriter.h"

#include <string>

void AddFuncKernel(IScriptEnvironment2* env);
void AddFuncMV(IScriptEnvironment2* env);

static void init_console()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONIN$", "r", stdin);
}

void OnCudaError(cudaError_t err) {
	printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
}

DeviceLocalBase::DeviceLocalBase(const void* init_data, size_t length, IScriptEnvironment2* env)
  : length(length)
{
  numDevices = env->GetProperty(AEP_NUM_DEVICES);
  dataPtrs = new std::atomic<void*>[numDevices]();
  void* ptr = new uint8_t[length];
  memcpy(ptr, init_data, length);
  dataPtrs[0].store(ptr, std::memory_order_relaxed);
}

DeviceLocalBase::~DeviceLocalBase()
{
  delete [] dataPtrs[0].load(std::memory_order_relaxed);
  for (int i = 1; i < numDevices; ++i) {
    void* ptr = dataPtrs[i].load(std::memory_order_relaxed);
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
  delete[] dataPtrs;
}

void* DeviceLocalBase::GetData_(IScriptEnvironment2* env)
{
  // double checked locking pattern
  int devid = env->GetProperty(AEP_DEVICE_ID);
  void* ptr = dataPtrs[devid].load(std::memory_order_acquire);
  if (ptr) return ptr;

  std::lock_guard<std::mutex> lock(mutex);
  ptr = dataPtrs[devid].load(std::memory_order_relaxed);
  if (ptr) return ptr;

  CUDA_CHECK(cudaMalloc(&ptr, length));
  CUDA_CHECK(cudaMemcpy(ptr, dataPtrs[0], length, cudaMemcpyHostToDevice));
  dataPtrs[devid].store(ptr, std::memory_order_release);

  return ptr;
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

  const int thresh;

  template <typename pixel_t>
  void ComparePlanar(pixel_t* d, const pixel_t* a, const pixel_t* b, int width, int height, int pitch, IScriptEnvironment2* env)
  {
    //DebugWriteBitmap("bob-ref-%d.bmp", (const uint8_t*)a, width, height, pitch, 1);
    //DebugWriteBitmap("bob-cuda-%d.bmp", (const uint8_t*)b, width, height, pitch, 1);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int off = x + y * pitch;
        pixel_t diff = std::abs(a[off] - b[off]);
        if (diff > thresh) {
          printf("miss match %d vs %d at (%d,%d)\n", a[off], b[off], x, y);
          env->ThrowError("[ImageCompare] âÊëúÇ™àÍívÇµÇ‹ÇπÇÒÅBÉeÉXÉgé∏îs");
        }
        d[off] = diff;
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

    ComparePlanar<pixel_t>(dstY, f1Y, f2Y,
      vi.width, vi.height, dst->GetPitch(PLANAR_Y) >> nPixelShift, env);
    ComparePlanar<pixel_t>(dstU, f1U, f2U,
      vi.width >> nUVShiftX, vi.height >> nUVShiftY, dst->GetPitch(PLANAR_U) >> nPixelShift, env);
    ComparePlanar<pixel_t>(dstV, f1V, f2V,
      vi.width >> nUVShiftX, vi.height >> nUVShiftY, dst->GetPitch(PLANAR_V) >> nPixelShift, env);
  }

public:
  ImageCompare(PClip child1, PClip child2, int thresh, IScriptEnvironment2* env)
    : GenericVideoFilter(child1)
    , child2(child2)
    , thresh(thresh)
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
      env);
  }
};

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env_, const AVS_Linkage* const vectors)
{
  IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
	AVS_linkage = vectors;
	//init_console();

  env->AddFunction("ImageCompare", "cc[thresh]i", ImageCompare::Create, 0);

	AddFuncKernel(env);
  AddFuncMV(env);

	return "CUDA Accelerated QTGMC Plugin";
}
