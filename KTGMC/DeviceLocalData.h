#pragma once

#include <avisynth.h>
#include <mutex>
#include <atomic>

class DeviceLocalBase
{
protected:
  DeviceLocalBase(const void* init_data, size_t length, IScriptEnvironment2* env);
  virtual ~DeviceLocalBase();

  std::atomic<void*>* dataPtrs;
  int numDevices;
  size_t length;
  std::mutex mutex;

  void* GetData_(IScriptEnvironment2* env);
};

template <typename T>
class DeviceLocalData : protected DeviceLocalBase
{
public:
  DeviceLocalData(const T* init_data, int size, IScriptEnvironment2* env)
    : DeviceLocalBase(init_data, size * sizeof(T), env)
  { }

  T* GetData(IScriptEnvironment2* env) {
    return (T*)GetData_(env);
  }
};
