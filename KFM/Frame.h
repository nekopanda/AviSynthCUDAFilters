#pragma once

#include <stdint.h>
#include <avisynth.h>


// ROIを持ったPVideoFrameのラッパ
struct Frame {
  PVideoFrame frame;
  // 全てバイト単位
  int offsetX, offsetY, offsetUVx, offsetUVy;
  int width, height, widthUV, heightUV;
  Frame() { }
  Frame(const PVideoFrame& frame)
    : frame(frame)
    , offsetX(0)
    , offsetY(0)
    , width(0)
    , height(0)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int cropY)
    : frame(frame)
    , offsetX(0)
    , offsetY(cropY)
    , width(frame->GetRowSize() - offsetX * 2)
    , height(frame->GetHeight() - offsetY * 2)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int cropX, int cropY, int pixelsize)
    : frame(frame)
    , offsetX(cropX * pixelsize)
    , offsetY(cropY)
    , width(frame->GetRowSize() - offsetX * 2)
    , height(frame->GetHeight() - offsetY * 2)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int offsetX, int offsetY, int width, int height, int pixelsize)
    : frame(frame)
    , offsetX(offsetX * pixelsize)
    , offsetY(offsetY)
    , width(width * pixelsize)
    , height(height)
  {
    init();
  }

  // for conditional expressions
  operator void*() const { return frame; }
  bool operator!() const { return !frame; }

  template <typename T> int GetPitch(int plane = 0) const {
    return frame->GetPitch(plane) / sizeof(T);
  }
  int GetRowSize(int plane = 0) const {
    return (plane & (PLANAR_U | PLANAR_V)) ? widthUV : width;
  }
  int GetHeight(int plane = 0) const {
    return (plane & (PLANAR_U | PLANAR_V)) ? heightUV : height;
  }
  template <typename T> int GetWidth(int plane = 0) const {
    return GetRowSize(plane) / sizeof(T);
  }
  template <typename T> const T* GetReadPtr(int plane = 0) {
    const BYTE* ptr = frame->GetReadPtr(plane);
    if (ptr) {
      ptr += (plane & (PLANAR_U | PLANAR_V))
        ? (offsetUVx + offsetUVy * widthUV)
        : (offsetX + offsetY * width);
    }
    return reinterpret_cast<const T*>(ptr);
  }
  template <typename T> T* GetWritePtr(int plane = 0) {
    BYTE* ptr = frame->GetWritePtr(plane);
    if (ptr) {
      ptr += (plane & (PLANAR_U | PLANAR_V))
        ? (offsetUVx + offsetUVy * widthUV)
        : (offsetX + offsetY * width);
    }
    return reinterpret_cast<T*>(ptr);
  }

  void SetProperty(const char* key, const AVSMapValue& value) { frame->SetProperty(key, value); }
  const AVSMapValue* GetProperty(const char* key) const { return frame->GetProperty(key); }
  PDevice GetDevice() const { return frame->GetDevice(); }
  int CheckMemory() const { return frame->CheckMemory(); }

  void Crop(int x, int y, int pixelsize) {
    offsetX += x * pixelsize;
    offsetY += y;
    offsetUVx += x * pixelsize >> ((widthUV < width) ? 1 : 0);
    offsetUVy += y >> ((heightUV < height) ? 1 : 0);
    widthUV -= x * pixelsize * ((widthUV < width) ? 1 : 2);
    heightUV -= y * ((heightUV < height) ? 1 : 2);
    width -= x * pixelsize * 2;
    height -= y * 2;
  }

private:
  void init()
  {
    if (width == 0) {
      width = frame->GetRowSize() - offsetX;
    }
    if (height == 0) {
      height = frame->GetHeight() - offsetY;
    }
    if (frame->GetRowSize(PLANAR_U) < frame->GetRowSize()) {
      // UVは横半分
      widthUV = width / 2;
      offsetUVx = offsetX / 2;
    }
    else {
      widthUV = width;
      offsetUVx = offsetX;
    }
    if (frame->GetHeight(PLANAR_U) < frame->GetHeight()) {
      // UVは縦半分
      heightUV = height / 2;
      offsetUVy = offsetY / 2;
    }
    else {
      heightUV = height;
      offsetUVy = offsetY;
    }
  }
};
