
#define NOMINMAX
#include <Windows.h>
#include "DebugWriter.h"

#include <stdio.h>
#include <string>
#include <memory>

bool FileExists(const char *fname)
{
  FILE *file;
  if (file = fopen(fname, "r")) {
    fclose(file);
    return true;
  }
  return false;
}

static std::string GetUniqueFileName(const char* fmt)
{
  char buf[200];
  for (int i = 0; i < 1000; ++i) {
    snprintf(buf, sizeof(buf), fmt, i);
    if (!FileExists(buf)) {
      return buf;
    }
  }
  return "error";
}

void DebugWrite8bitColorBitmap(const char* filenamefmt, const void* data, int width, int height)
{
  BITMAPFILEHEADER	bmpFileHeader = { 0 };
  BITMAPINFOHEADER	bmpInfoHeader = { 0 };

  bmpFileHeader.bfType = 0x4d42;	/* "BM" */
  bmpFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
  bmpFileHeader.bfSize = width * height * 3 + bmpFileHeader.bfOffBits;

  bmpInfoHeader.biBitCount = 24;
  bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmpInfoHeader.biWidth = width;
  bmpInfoHeader.biHeight = height;
  bmpInfoHeader.biPlanes = 1;
  bmpInfoHeader.biCompression = BI_RGB;

  auto filename = GetUniqueFileName(filenamefmt);
  FILE *fp = fopen(filename.c_str(), "wb");
  fwrite(&bmpFileHeader, sizeof(bmpFileHeader), 1, fp);
  fwrite(&bmpInfoHeader, sizeof(bmpInfoHeader), 1, fp);
  fwrite(data, 1, width * height * 3, fp);
  fclose(fp);
  printf("DebugWriteBitmap -> %s\n", filename.c_str());
}

void DebugWriteGrayBitmap(const char* filenamefmt, const void* data, int width, int height, int pixelSize)
{
  BITMAPFILEHEADER	bmpFileHeader = { 0 };
  BITMAPINFOHEADER	bmpInfoHeader = { 0 };

  bmpFileHeader.bfType = 0x4d42;	/* "BM" */
  bmpFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
  bmpFileHeader.bfSize = width * height + bmpFileHeader.bfOffBits;

  bmpInfoHeader.biBitCount = pixelSize * 8;
  bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmpInfoHeader.biWidth = width;
  bmpInfoHeader.biHeight = height;
  bmpInfoHeader.biPlanes = 1;
  bmpInfoHeader.biCompression = BI_RGB;

  auto filename = GetUniqueFileName(filenamefmt);
  FILE *fp = fopen(filename.c_str(), "wb");
  fwrite(&bmpFileHeader, sizeof(bmpFileHeader), 1, fp);
  fwrite(&bmpInfoHeader, sizeof(bmpInfoHeader), 1, fp);
  fwrite(data, width * height * pixelSize, 1, fp);
  fclose(fp);
  printf("DebugWriteBitmap -> %s\n", filename.c_str());
}

void DebugWriteBitmap(const char* filenamefmt, const uint8_t* Y, int width, int height, int strideY, int pixelSize)
{
  std::unique_ptr<uint8_t[]> tmpbuf = std::unique_ptr<uint8_t[]>(new uint8_t[width*height*3]);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      tmpbuf[x * 3 + 0 + y * width * 3] = Y[x + (height - y - 1) * strideY];
      tmpbuf[x * 3 + 1 + y * width * 3] = Y[x + (height - y - 1) * strideY];
      tmpbuf[x * 3 + 2 + y * width * 3] = Y[x + (height - y - 1) * strideY];
    }
  }
  DebugWrite8bitColorBitmap(filenamefmt, tmpbuf.get(), width, height);
}

void DebugWriteBitmap(const char* filenamefmt, const uint8_t* Y, const uint8_t* U, const uint8_t* V, int width, int height, int strideY, int strideUV, int pixelSize)
{
  // TODO:
}
