
#include "DeviceLocalData.h"

DeviceLocalBase::DeviceLocalBase(const void* init_data, size_t length, IScriptEnvironment2* env)
  : length(length)
{
  numDevices = (int)env->GetProperty(AEP_NUM_DEVICES);
  dataPtrs = new std::atomic<void*>[numDevices]();
  void* ptr = new uint8_t[length];
  memcpy(ptr, init_data, length);
  dataPtrs[0].store(ptr, std::memory_order_relaxed);
}

DeviceLocalBase::~DeviceLocalBase()
{
  delete[] dataPtrs[0].load(std::memory_order_relaxed);
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
  int devid = (int)env->GetProperty(AEP_DEVICE_ID);
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
