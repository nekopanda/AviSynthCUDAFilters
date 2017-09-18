#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "avisynth.h"

#undef min
#undef max

#include "CommonFunctions.h"
#include "DeviceLocalData.h"

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

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env_, const AVS_Linkage* const vectors)
{
  IScriptEnvironment2* env = static_cast<IScriptEnvironment2*>(env_);
	AVS_linkage = vectors;
	//init_console();

	AddFuncKernel(env);
  AddFuncMV(env);

	return "CUDA Accelerated Deinterlace Plugin";
}
