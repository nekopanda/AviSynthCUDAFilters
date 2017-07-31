#pragma once

#include "CommonFunctions.h"

#include <math.h>

class OverlapWindows
{
  int nx; // window sizes
  int ny;
  int ox; // overap sizes
  int oy;
  int size; // full window size= nx*ny

  short * Overlap9Windows;

  float *fWin1UVx;
  float *fWin1UVxfirst;
  float *fWin1UVxlast;
  float *fWin1UVy;
  float *fWin1UVyfirst;
  float *fWin1UVylast;
public:

  OverlapWindows(int _nx, int _ny, int _ox, int _oy)
  {
    nx = _nx;
    ny = _ny;
    ox = _ox;
    oy = _oy;
    size = nx*ny;

    //  windows
    fWin1UVx = new float[nx];
    fWin1UVxfirst = new float[nx];
    fWin1UVxlast = new float[nx];
    for (int i = 0; i<ox; i++)
    {
      fWin1UVx[i] = float(cos(PI*(i - ox + 0.5f) / (ox * 2)));
      fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];// left window (rised cosine)
      fWin1UVxfirst[i] = 1; // very first window
      fWin1UVxlast[i] = fWin1UVx[i]; // very last
    }
    for (int i = ox; i<nx - ox; i++)
    {
      fWin1UVx[i] = 1;
      fWin1UVxfirst[i] = 1; // very first window
      fWin1UVxlast[i] = 1; // very last
    }
    for (int i = nx - ox; i<nx; i++)
    {
      fWin1UVx[i] = float(cos(PI*(i - nx + ox + 0.5f) / (ox * 2)));
      fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];// right window (falled cosine)
      fWin1UVxfirst[i] = fWin1UVx[i]; // very first window
      fWin1UVxlast[i] = 1; // very last
    }

    fWin1UVy = new float[ny];
    fWin1UVyfirst = new float[ny];
    fWin1UVylast = new float[ny];
    for (int i = 0; i<oy; i++)
    {
      fWin1UVy[i] = float(cos(PI*(i - oy + 0.5f) / (oy * 2)));
      fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];// left window (rised cosine)
      fWin1UVyfirst[i] = 1; // very first window
      fWin1UVylast[i] = fWin1UVy[i]; // very last
    }
    for (int i = oy; i<ny - oy; i++)
    {
      fWin1UVy[i] = 1;
      fWin1UVyfirst[i] = 1; // very first window
      fWin1UVylast[i] = 1; // very last
    }
    for (int i = ny - oy; i<ny; i++)
    {
      fWin1UVy[i] = float(cos(PI*(i - ny + oy + 0.5f) / (oy * 2)));
      fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];// right window (falled cosine)
      fWin1UVyfirst[i] = fWin1UVy[i]; // very first window
      fWin1UVylast[i] = 1; // very last
    }


    Overlap9Windows = new short[size * 9];

    short *winOverUVTL = Overlap9Windows;
    short *winOverUVTM = Overlap9Windows + size;
    short *winOverUVTR = Overlap9Windows + size * 2;
    short *winOverUVML = Overlap9Windows + size * 3;
    short *winOverUVMM = Overlap9Windows + size * 4;
    short *winOverUVMR = Overlap9Windows + size * 5;
    short *winOverUVBL = Overlap9Windows + size * 6;
    short *winOverUVBM = Overlap9Windows + size * 7;
    short *winOverUVBR = Overlap9Windows + size * 8;

    for (int j = 0; j<ny; j++)
    {
      for (int i = 0; i<nx; i++)
      {
        winOverUVTL[i] = (int)(fWin1UVyfirst[j] * fWin1UVxfirst[i] * 2048 + 0.5f);
        winOverUVTM[i] = (int)(fWin1UVyfirst[j] * fWin1UVx[i] * 2048 + 0.5f);
        winOverUVTR[i] = (int)(fWin1UVyfirst[j] * fWin1UVxlast[i] * 2048 + 0.5f);
        winOverUVML[i] = (int)(fWin1UVy[j] * fWin1UVxfirst[i] * 2048 + 0.5f);
        winOverUVMM[i] = (int)(fWin1UVy[j] * fWin1UVx[i] * 2048 + 0.5f);
        winOverUVMR[i] = (int)(fWin1UVy[j] * fWin1UVxlast[i] * 2048 + 0.5f);
        winOverUVBL[i] = (int)(fWin1UVylast[j] * fWin1UVxfirst[i] * 2048 + 0.5f);
        winOverUVBM[i] = (int)(fWin1UVylast[j] * fWin1UVx[i] * 2048 + 0.5f);
        winOverUVBR[i] = (int)(fWin1UVylast[j] * fWin1UVxlast[i] * 2048 + 0.5f);
      }
      winOverUVTL += nx;
      winOverUVTM += nx;
      winOverUVTR += nx;
      winOverUVML += nx;
      winOverUVMM += nx;
      winOverUVMR += nx;
      winOverUVBL += nx;
      winOverUVBM += nx;
      winOverUVBR += nx;
    }
  }

  ~OverlapWindows()
  {
    delete[] Overlap9Windows;
    delete[] fWin1UVx;
    delete[] fWin1UVxfirst;
    delete[] fWin1UVxlast;
    delete[] fWin1UVy;
    delete[] fWin1UVyfirst;
    delete[] fWin1UVylast;
  }

  int Getnx() const { return nx; }
  int Getny() const { return ny; }
  int GetSize() const { return size; }
  short *GetWindow(int i) const { return Overlap9Windows + size*i; }

  const short* GetWindow(int bx, int by, int numBlkX, int numBlkY) const {
    int wbx = (bx + numBlkX - 3) / (numBlkX - 2);
    int wby = (by + numBlkY - 3) / (numBlkY - 2);
    return &Overlap9Windows[size * (wbx + wby * 3)];
  }
};
