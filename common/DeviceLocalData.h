#pragma once

#include <avisynth.h>
#include <mutex>
#include <atomic>

class DeviceLocalBase
{
protected:
  DeviceLocalBase(const void* init_data, size_t length, PNeoEnv env);
  virtual ~DeviceLocalBase();

  std::atomic<void*>* dataPtrs;
  int numDevices;
  size_t length;
  std::mutex mutex;

  void* GetData_(PNeoEnv env);
};

template <typename T>
class DeviceLocalData : protected DeviceLocalBase
{
public:
  DeviceLocalData(const T* init_data, int size, PNeoEnv env)
    : DeviceLocalBase(init_data, size * sizeof(T), env)
  { }

  T* GetData(PNeoEnv env) {
    return (T*)GetData_(env);
  }
};
