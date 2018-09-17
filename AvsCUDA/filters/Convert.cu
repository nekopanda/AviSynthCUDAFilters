// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.


#include "../AvsCUDA.h"
#include <avs/alignment.h>
#include <avs/win.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <tuple>
#include <map>

#include "convert_avx.h"
#include "convert_avx2.h"

#include "Copy.h"

//--------------- planar bit depth conversions
// todo: separate file?
typedef void(*BitDepthConvFuncPtr)(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
typedef void(*BitDepthConvFuncPtr2)(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env);

class ConvertBits : public GenericVideoFilter
{
public:
	ConvertBits(PClip _child, const int _dither_mode, const int _target_bitdepth, bool _truerange, bool _fulls, bool _fulld, int _dither_bitdepth, IScriptEnvironment* env);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

	int __stdcall SetCacheHints(int cachehints, int frame_range) override {
		if (cachehints == CACHE_GET_DEV_TYPE) {
			return GetDeviceTypes(child) &
				(DEV_TYPE_CPU | DEV_TYPE_CUDA);
		}
		return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
	}

	static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
	BitDepthConvFuncPtr conv_function;
	BitDepthConvFuncPtr conv_function_chroma; // 32bit float YUV chroma
	BitDepthConvFuncPtr conv_function_a;
	BitDepthConvFuncPtr2 conv_cuda;
	BitDepthConvFuncPtr2 conv_cuda_chroma;
	float float_range;
	int dither_mode;
	int pixelsize;
	int bits_per_pixel;
	int target_bitdepth;
	int dither_bitdepth;
	bool truerange; // if 16->10 range reducing or e.g. 14->16 bit range expansion needed
	bool format_change_only;
	bool fulls; // source is full range (defaults: rgb=true, yuv=false (bit shift))
	bool fulld; // destination is full range (defaults: rgb=true, yuv=false (bit shift))
};

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const FuncDefinition Convert_filters[] = {       // matrix can be "rec601", "rec709", "PC.601" or "PC.709" or "rec2020"
	{ "ConvertTo8bit",  BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]f[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)8 },
	{ "ConvertTo16bit", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]f[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)16 },
	{ "ConvertToFloat", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]f[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)32 },
	{ "ConvertBits",    BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]f[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)0 },
	{ 0 }
};

template <int SHIFT> struct Dither {
	static __device__ BYTE get(int x, int y) { return 0; }
};

__constant__ BYTE c_dither2[2][2] = {
	{ 0, 2 },
	{ 3, 1 }
};
template <> struct Dither<2> {
	static __device__ BYTE get(int x, int y) { return c_dither2[y][x]; }
};

__constant__ BYTE c_dither4[4][4] = {
	{ 0,  8,  2, 10 },
	{ 12,  4, 14,  6 },
	{ 3, 11,  1,  9 },
	{ 15,  7, 13,  5 }
};
template <> struct Dither<4> {
	static __device__ BYTE get(int x, int y) { return c_dither4[y][x]; }
};

__constant__ BYTE c_dither6[8][8] = {
	{ 0, 32,  8, 40,  2, 34, 10, 42 }, /* 8x8 Bayer ordered dithering */
	{ 48, 16, 56, 24, 50, 18, 58, 26 }, /* pattern. Each input pixel */
	{ 12, 44,  4, 36, 14, 46,  6, 38 }, /* is scaled to the 0..63 range */
	{ 60, 28, 52, 20, 62, 30, 54, 22 }, /* before looking in this table */
	{ 3, 35, 11, 43,  1, 33,  9, 41 }, /* to determine the action. */
	{ 51, 19, 59, 27, 49, 17, 57, 25 },
	{ 15, 47,  7, 39, 13, 45,  5, 37 },
	{ 63, 31, 55, 23, 61, 29, 53, 21 }
};
template <> struct Dither<6> {
	static __device__ BYTE get(int x, int y) { return c_dither6[y][x]; }
};

__constant__ BYTE c_dither8[16][16] = {
	{ 0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
	{ 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
	{ 32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
	{ 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
	{ 8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
	{ 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
	{ 40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
	{ 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
	{ 2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
	{ 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
	{ 34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
	{ 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
	{ 10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
	{ 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
	{ 42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
	{ 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
};
template <> struct Dither<8> {
	static __device__ BYTE get(int x, int y) { return c_dither8[y][x]; }
};

template <int TYPE>
struct ConvertBitsKernel { };

template <int BITS> struct BitsToType { typedef uint16_t type; };
template <> struct BitsToType<8> { typedef uint8_t type; };
template <> struct BitsToType<32> { typedef float type; };

// 0: copy
template <> struct ConvertBitsKernel<0> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		Copy(dstp, dst_pitch, srcp, src_pitch, src_rowsize, src_height, env);
	}
};

template <typename SRC_TYPE, typename TGT_TYPE, int SHIFT, int TGT_BITS>
__global__ void kl_convert_to_lower_bits_dither(int width, int height,
	TGT_TYPE* dst, int dst_pitch, const SRC_TYPE* src, int src_pitch)
{
	enum { 
		DITHER_W = (1 << (SHIFT >> 1)), 
		MASK = DITHER_W - 1
	};

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		auto tmp = (src[x + y * src_pitch] + Dither<SHIFT>::get(x & MASK, y & MASK)) >> SHIFT;
		dst[x + y * dst_pitch] = (TGT_TYPE)min(tmp, (1 << TGT_BITS) - 1);
	}
}

// 1: convert_to_lower_bits_dither
template <> struct ConvertBitsKernel<1> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		typedef typename BitsToType<SRC_BITS>::type SRC_TYPE;
		typedef typename BitsToType<TGT_BITS>::type TGT_TYPE;
		int width = src_rowsize / sizeof(SRC_TYPE);
		int height = src_height;
		dim3 threads(32, 16);
		dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
		kl_convert_to_lower_bits_dither
			<SRC_TYPE, TGT_TYPE, SRC_BITS - TGT_BITS, TGT_BITS>
			<< <blocks, threads >> > (
				width, height,
				(TGT_TYPE*)dstp, dst_pitch / sizeof(TGT_TYPE),
				(SRC_TYPE*)srcp, src_pitch / sizeof(SRC_TYPE));
		DEBUG_SYNC;
	}
};

template <typename SRC_TYPE, typename TGT_TYPE, int SHIFT, int TGT_BITS>
__global__ void kl_convert_to_lower_bits_no_dither(int width, int height,
	TGT_TYPE* dst, int dst_pitch, const SRC_TYPE* src, int src_pitch)
{
	enum { 
		HALF = (1 << (SHIFT - 1))
	};

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		//auto tmp = (src[x + y * src_pitch] + HALF) >> SHIFT;
		auto tmp = (src[x + y * src_pitch]) >> SHIFT;
		dst[x + y * dst_pitch] = (TGT_TYPE)min(tmp, (1 << TGT_BITS) - 1);
	}
}

// 2: convert_to_lower_bits_no_dither
template <> struct ConvertBitsKernel<2> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		typedef typename BitsToType<SRC_BITS>::type SRC_TYPE;
		typedef typename BitsToType<TGT_BITS>::type TGT_TYPE;
		int width = src_rowsize / sizeof(SRC_TYPE);
		int height = src_height;
		dim3 threads(32, 16);
		dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
		kl_convert_to_lower_bits_no_dither
			<SRC_TYPE, TGT_TYPE, SRC_BITS - TGT_BITS, TGT_BITS>
			<< <blocks, threads >> > (
				width, height,
				(TGT_TYPE*)dstp, dst_pitch / sizeof(TGT_TYPE),
				(SRC_TYPE*)srcp, src_pitch / sizeof(SRC_TYPE));
		DEBUG_SYNC;
	}
};

template <typename SRC_TYPE, typename TGT_TYPE, int SHIFT, int TGT_BITS>
__global__ void kl_convert_to_higher_bits(int width, int height, 
	TGT_TYPE* dst, int dst_pitch, const SRC_TYPE* src, int src_pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {
		dst[x + y * dst_pitch] = (TGT_TYPE)min(src[x + y * src_pitch] << SHIFT, (1 << TGT_BITS) - 1);
	}
}

// 3: convert_to_higher_bits
template <> struct ConvertBitsKernel<3> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		typedef typename BitsToType<SRC_BITS>::type SRC_TYPE;
		typedef typename BitsToType<TGT_BITS>::type TGT_TYPE;
		int width = src_rowsize / sizeof(SRC_TYPE);
		int height = src_height;
		dim3 threads(32, 16);
		dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
		kl_convert_to_higher_bits
			<SRC_TYPE, TGT_TYPE, TGT_BITS - SRC_BITS, TGT_BITS>
			<< <blocks, threads >> > (
				width, height,
				(TGT_TYPE*)dstp, dst_pitch / sizeof(TGT_TYPE),
				(SRC_TYPE*)srcp, src_pitch / sizeof(SRC_TYPE));
		DEBUG_SYNC;
	}
};

template <typename TGT_TYPE, int TGT_BITS, bool CHROMA>
__global__ void kl_convert_from_float(int width, int height,
	TGT_TYPE* dst, int dst_pitch, const float* src, int src_pitch)
{
	constexpr float MAX_VAL = 255 << (TGT_BITS - 8);
	constexpr float HALF = 128 << (TGT_BITS - 8);
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		float tmp;
		if (CHROMA) {
			tmp = src[x + y * src_pitch] * MAX_VAL + HALF + 0.5f;
		}
		else {
			tmp = src[x + y * src_pitch] * MAX_VAL + 0.5f;
		}
		dst[x + y * dst_pitch] = (TGT_TYPE)clamp(tmp, 0.0f, (float)MAX_VAL);
	}
}

// 4: convert_from_float
template <> struct ConvertBitsKernel<4> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		typedef float SRC_TYPE;
		typedef typename BitsToType<TGT_BITS>::type TGT_TYPE;
		int width = src_rowsize / sizeof(SRC_TYPE);
		int height = src_height;
		dim3 threads(32, 16);
		dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
		kl_convert_from_float
			<TGT_TYPE, TGT_BITS, CHROMA>
			<< <blocks, threads >> > (
				width, height,
				(TGT_TYPE*)dstp, dst_pitch / sizeof(TGT_TYPE),
				(SRC_TYPE*)srcp, src_pitch / sizeof(SRC_TYPE));
		DEBUG_SYNC;
	}
};

template <typename SRC_TYPE, int SRC_BITS, bool CHROMA>
__global__ void kl_convert_to_float(int width, int height,
	float* dst, int dst_pitch, const SRC_TYPE* src, int src_pitch)
{
	constexpr float MAX_VAL = 255 << (SRC_BITS - 8);
	constexpr float FACTOR = 1.0f / MAX_VAL;
	constexpr float HALF = 128 << (SRC_BITS - 8);
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		float tmp;
		if (CHROMA) {
			tmp = (float)(src[x + y * src_pitch] - HALF) * FACTOR;
		}
		else {
			tmp = (float)src[x + y * src_pitch] * FACTOR;
		}
		dst[x + y * dst_pitch] = tmp;
	}
}

// 5: convert_to_float
template <> struct ConvertBitsKernel<5> {
	template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
	static void launch(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, PNeoEnv env)
	{
		typedef typename BitsToType<SRC_BITS>::type SRC_TYPE;
		typedef float TGT_TYPE;
		int width = src_rowsize / sizeof(SRC_TYPE);
		int height = src_height;
		dim3 threads(32, 16);
		dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
		kl_convert_to_float
			<SRC_TYPE, SRC_BITS, CHROMA>
			<< <blocks, threads >> > (
				width, height,
				(TGT_TYPE*)dstp, dst_pitch / sizeof(TGT_TYPE),
				(SRC_TYPE*)srcp, src_pitch / sizeof(SRC_TYPE));
		DEBUG_SYNC;
	}
};


/**********************************
******  Bitdepth conversions  *****
**********************************/
// 10->8
// repeated 4x for sse size 16
static const struct dither2x2_t
{
	const BYTE data[4] = {
		0, 2,
		3, 1,
	};
	// cycle: 2
	alignas(16) const BYTE data_sse2[2 * 16] = {
		0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
		3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1
	};
	dither2x2_t() {};
} dither2x2;


// 12->8
static const struct dither4x4_t
{
	const BYTE data[16] = {
		0,  8,  2, 10,
		12,  4, 14,  6,
		3, 11,  1,  9,
		15,  7, 13,  5
	};
	// cycle: 4
	alignas(16) const BYTE data_sse2[4 * 16] = {
		0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,  0,  8,  2, 10,
		12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6, 12,  4, 14,  6,
		3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,  3, 11,  1,  9,
		15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5, 15,  7, 13,  5
	};
	dither4x4_t() {};
} dither4x4;

// 14->8
static const struct dither8x8_t
{
	const BYTE data[8][8] = {
		{ 0, 32,  8, 40,  2, 34, 10, 42 }, /* 8x8 Bayer ordered dithering */
		{ 48, 16, 56, 24, 50, 18, 58, 26 }, /* pattern. Each input pixel */
		{ 12, 44,  4, 36, 14, 46,  6, 38 }, /* is scaled to the 0..63 range */
		{ 60, 28, 52, 20, 62, 30, 54, 22 }, /* before looking in this table */
		{ 3, 35, 11, 43,  1, 33,  9, 41 }, /* to determine the action. */
		{ 51, 19, 59, 27, 49, 17, 57, 25 },
		{ 15, 47,  7, 39, 13, 45,  5, 37 },
		{ 63, 31, 55, 23, 61, 29, 53, 21 }
	};
	// cycle: 8
	alignas(16) const BYTE data_sse2[8][16] = {
		{ 0, 32,  8, 40,  2, 34, 10, 42,  0, 32,  8, 40,  2, 34, 10, 42 }, /* 8x8 Bayer ordered dithering */
		{ 48, 16, 56, 24, 50, 18, 58, 26, 48, 16, 56, 24, 50, 18, 58, 26 }, /* pattern. Each input pixel */
		{ 12, 44,  4, 36, 14, 46,  6, 38, 12, 44,  4, 36, 14, 46,  6, 38 }, /* is scaled to the 0..63 range */
		{ 60, 28, 52, 20, 62, 30, 54, 22, 60, 28, 52, 20, 62, 30, 54, 22 }, /* before looking in this table */
		{ 3, 35, 11, 43,  1, 33,  9, 41,  3, 35, 11, 43,  1, 33,  9, 41 }, /* to determine the action. */
		{ 51, 19, 59, 27, 49, 17, 57, 25, 51, 19, 59, 27, 49, 17, 57, 25 },
		{ 15, 47,  7, 39, 13, 45,  5, 37, 15, 47,  7, 39, 13, 45,  5, 37 },
		{ 63, 31, 55, 23, 61, 29, 53, 21, 63, 31, 55, 23, 61, 29, 53, 21 }
	};
	dither8x8_t() {};
} dither8x8;

// 16->8
static const struct dither16x16_t
{
	// cycle: 16x
	alignas(16) const BYTE data[16][16] = {
		{ 0,192, 48,240, 12,204, 60,252,  3,195, 51,243, 15,207, 63,255 },
		{ 128, 64,176,112,140, 76,188,124,131, 67,179,115,143, 79,191,127 },
		{ 32,224, 16,208, 44,236, 28,220, 35,227, 19,211, 47,239, 31,223 },
		{ 160, 96,144, 80,172,108,156, 92,163, 99,147, 83,175,111,159, 95 },
		{ 8,200, 56,248,  4,196, 52,244, 11,203, 59,251,  7,199, 55,247 },
		{ 136, 72,184,120,132, 68,180,116,139, 75,187,123,135, 71,183,119 },
		{ 40,232, 24,216, 36,228, 20,212, 43,235, 27,219, 39,231, 23,215 },
		{ 168,104,152, 88,164,100,148, 84,171,107,155, 91,167,103,151, 87 },
		{ 2,194, 50,242, 14,206, 62,254,  1,193, 49,241, 13,205, 61,253 },
		{ 130, 66,178,114,142, 78,190,126,129, 65,177,113,141, 77,189,125 },
		{ 34,226, 18,210, 46,238, 30,222, 33,225, 17,209, 45,237, 29,221 },
		{ 162, 98,146, 82,174,110,158, 94,161, 97,145, 81,173,109,157, 93 },
		{ 10,202, 58,250,  6,198, 54,246,  9,201, 57,249,  5,197, 53,245 },
		{ 138, 74,186,122,134, 70,182,118,137, 73,185,121,133, 69,181,117 },
		{ 42,234, 26,218, 38,230, 22,214, 41,233, 25,217, 37,229, 21,213 },
		{ 170,106,154, 90,166,102,150, 86,169,105,153, 89,165,101,149, 85 }
	};
	dither16x16_t() {};
} dither16x16;

template<uint8_t sourcebits, int dither_mode, int TARGET_DITHER_BITDEPTH, int rgb_step>
static void convert_rgb_uint16_to_8_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
	src_pitch = src_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);

	int _y = 0; // for ordered dither

	const int TARGET_BITDEPTH = 8; // here is constant (uint8_t target)

																 // for test, make it 2,4,6,8. sourcebits-TARGET_DITHER_BITDEPTH cannot exceed 8 bit
																 // const int TARGET_DITHER_BITDEPTH = 2;

	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1;
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but generality)
	const int DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const int DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const int MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data); break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data); break;
	case 6: matrix = reinterpret_cast<const BYTE *>(dither8x8.data); break;
	case 8: matrix = reinterpret_cast<const BYTE *>(dither16x16.data); break;
  default: return; // n/a
	}

	for (int y = 0; y<src_height; y++)
	{
    if constexpr(dither_mode == 0)
			_y = (y & MASK) << DITHER_ORDER; // ordered dither
		for (int x = 0; x < src_width; x++)
		{
      if constexpr(dither_mode < 0) // -1: no dither
			{
				const float mulfactor = sourcebits == 16 ? (1.0f / 257.0f) :
					sourcebits == 14 ? (255.0f / 16383.0f) :
					sourcebits == 12 ? (255.0f / 4095.0f) :
					(255.0f / 1023.0f); // 10 bits

				dstp[x] = (uint8_t)(srcp0[x] * mulfactor + 0.5f);
				// C cast truncates, use +0.5f rounder, which uses cvttss2si

				// old method: no rounding but fast
				// no integer division (fast tricky algorithm by compiler), rounding problems, pic gets darker
				// dstp[x] = srcp0[x] / 257; // RGB: full range 0..255 <-> 0..65535 (*255 / 65535)
				// dstp[x] = srcp0[x] * 255 / 16383; // RGB: full range 0..255 <-> 0..16384-1
				// dstp[x] = srcp0[x] * 255 / 4095; // RGB: full range 0..255 <-> 0..4096-1
				// dstp[x] = srcp0[x] * 255 / 1023; // RGB: full range 0..255 <-> 0..1024-1
			}
			else { // dither_mode == 0 -> ordered dither
				const int corr = matrix[_y | ((x / rgb_step) & MASK)];
				// vvv for the non-fullscale version: int new_pixel = ((srcp0[x] + corr) >> DITHER_BIT_DIFF);
				int new_pixel;

				const float mulfactor =
					DITHER_BIT_DIFF == 8 ? (1.0f / 257.0f) :
					DITHER_BIT_DIFF == 6 ? (255.0f / 16383.0f) :
					DITHER_BIT_DIFF == 4 ? (255.0f / 4095.0f) :
					DITHER_BIT_DIFF == 2 ? (255.0f / 1023.0f) : // 10 bits
					1.0f;

        if constexpr(TARGET_DITHER_BITDEPTH <= 4)
					new_pixel = (uint16_t)((srcp0[x] + corr) * mulfactor); // rounding here makes brightness shift
        else if constexpr(DITHER_BIT_DIFF > 0)
					new_pixel = (uint16_t)((srcp0[x] + corr) * mulfactor + 0.5f);
				else
					new_pixel = (uint16_t)(srcp0[x] + corr);

				new_pixel = min(new_pixel, max_pixel_value_dithered); // clamp upper

																															// scale back to the required bit depth
																															// for generality. Now target == 8 bit, and dither_target is also 8 bit
																															// for test: source:10 bit, target=8 bit, dither_target=4 bit
				const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
        if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0)  // dither to 8, target to 8
					new_pixel = new_pixel << BITDIFF_BETWEEN_DITHER_AND_TARGET; // if implemented non-8bit dither target, this should be fullscale
				dstp[x] = (BYTE)new_pixel;
			}
		} // x
		dstp += dst_pitch;
		srcp0 += src_pitch;
	}
}

template<uint8_t sourcebits, int dither_mode, int TARGET_DITHER_BITDEPTH, int rgb_step>
static void convert_rgb_uint16_to_8_sse2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
	src_pitch = src_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);

	int _y = 0; // for ordered dither

	const int TARGET_BITDEPTH = 8; // here is constant (uint8_t target)

																 // for test, make it 2,4,6,8. sourcebits-TARGET_DITHER_BITDEPTH cannot exceed 8 bit
																 // const int TARGET_DITHER_BITDEPTH = 2;

	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1;
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but generality)
	const int DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const int DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const int MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data); break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data); break;
	case 6: matrix = reinterpret_cast<const BYTE *>(dither8x8.data); break;
	case 8: matrix = reinterpret_cast<const BYTE *>(dither16x16.data); break;
  default: return; // n/a
	}

	// 20171024: given up integer division, rounding problems
	const float mulfactor =
		sourcebits == 16 ? (1.0f / 257.0f) :
		sourcebits == 14 ? (255.0f / 16383.0f) :
		sourcebits == 12 ? (255.0f / 4095.0f) :
		(255.0f / 1023.0f); // 10 bits
	const __m128 mulfactor_simd = _mm_set1_ps(mulfactor);
	const __m128i zero = _mm_setzero_si128();

	for (int y = 0; y < src_height; y++)
	{
    if constexpr(dither_mode == 0)
			_y = (y & MASK) << DITHER_ORDER; // ordered dither
		for (int x = 0; x < src_width; x += 8) // 8 * uint16_t at a time
		{
      if constexpr(dither_mode < 0) // -1: no dither
			{

				// C: dstp[x] = (uint8_t)(srcp0[x] * mulfactor + 0.5f);
				// C cast truncates, use +0.5f rounder, which uses cvttss2si

				__m128i pixel_i = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp0 + x)); // 16 bytes 8 pixels

				__m128 pixel_f_lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(pixel_i, zero)); // 4 floats
				__m128 mulled_lo = _mm_mul_ps(pixel_f_lo, mulfactor_simd);
				__m128i converted32_lo = _mm_cvtps_epi32(mulled_lo); // rounding ok, nearest. no +0.5 needed

				__m128 pixel_f_hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(pixel_i, zero)); // 4 floats
				__m128 mulled_hi = _mm_mul_ps(pixel_f_hi, mulfactor_simd);
				__m128i converted32_hi = _mm_cvtps_epi32(mulled_hi);

				__m128i converted_16 = _mm_packs_epi32(converted32_lo, converted32_hi);
				__m128i converted_8 = _mm_packus_epi16(converted_16, zero);
				_mm_storel_epi64(reinterpret_cast<__m128i *>(&dstp[x]), converted_8); // store 8 bytes
			}
			else { // dither_mode == 0 -> ordered dither
						 //  const int corr = matrix[_y | ((x / rgb_step) & MASK)];
				__m128i corr_lo = _mm_set_epi32(
					matrix[_y | (((x + 3) / rgb_step) & MASK)],
					matrix[_y | (((x + 2) / rgb_step) & MASK)],
					matrix[_y | (((x + 1) / rgb_step) & MASK)],
					matrix[_y | (((x + 0) / rgb_step) & MASK)]
				);
				__m128i corr_hi = _mm_set_epi32(
					matrix[_y | (((x + 7) / rgb_step) & MASK)],
					matrix[_y | (((x + 6) / rgb_step) & MASK)],
					matrix[_y | (((x + 5) / rgb_step) & MASK)],
					matrix[_y | (((x + 4) / rgb_step) & MASK)]
				);
				// vvv for the non-fullscale version: int new_pixel = ((srcp0[x] + corr) >> DITHER_BIT_DIFF);

				// no integer division, rounding problems
				const float mulfactor_dith =
					DITHER_BIT_DIFF == 8 ? (1.0f / 257.0f) :
					DITHER_BIT_DIFF == 6 ? (255.0f / 16383.0f) :
					DITHER_BIT_DIFF == 4 ? (255.0f / 4095.0f) :
					DITHER_BIT_DIFF == 2 ? (255.0f / 1023.0f) :
					1.0f;
				__m128 mulfactor_dith_simd = _mm_set1_ps(mulfactor_dith);

				__m128i pixel_i = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp0 + x)); // 16 bytes 8 pixels
				__m128i pixel_i_lo = _mm_add_epi32(_mm_unpacklo_epi16(pixel_i, zero), corr_lo);
				__m128i pixel_i_hi = _mm_add_epi32(_mm_unpackhi_epi16(pixel_i, zero), corr_hi);
				__m128i converted32_lo, converted32_hi;

				/* C:
				if (TARGET_DITHER_BITDEPTH <= 4)
				new_pixel = (uint16_t)((srcp0[x] + corr) * mulfactor); // rounding here makes brightness shift
				else if (DITHER_BIT_DIFF > 0)
				new_pixel = (uint16_t)((srcp0[x] + corr) * mulfactor + 0.5f);
				else
				new_pixel = (uint16_t)(srcp0[x] + corr);
				*/
        if constexpr(TARGET_DITHER_BITDEPTH <= 4) {
					// round: truncate
					__m128 pixel_f_lo = _mm_cvtepi32_ps(pixel_i_lo); // 4 floats
					__m128 mulled_lo = _mm_mul_ps(pixel_f_lo, mulfactor_dith_simd);
					converted32_lo = _mm_cvttps_epi32(mulled_lo); // truncate! rounding here makes brightness shift

					__m128 pixel_f_hi = _mm_cvtepi32_ps(pixel_i_hi); // 4 floats
					__m128 mulled_hi = _mm_mul_ps(pixel_f_hi, mulfactor_dith_simd);
					converted32_hi = _mm_cvttps_epi32(mulled_hi); // truncate! rounding here makes brightness shift
				}
        else if constexpr(DITHER_BIT_DIFF > 0) {
					// round: nearest
					__m128 pixel_f_lo = _mm_cvtepi32_ps(pixel_i_lo); // 4 floats
					__m128 mulled_lo = _mm_mul_ps(pixel_f_lo, mulfactor_dith_simd);
					converted32_lo = _mm_cvtps_epi32(mulled_lo); // rounding ok, nearest. no +0.5 needed

					__m128 pixel_f_hi = _mm_cvtepi32_ps(pixel_i_hi); // 4 floats
					__m128 mulled_hi = _mm_mul_ps(pixel_f_hi, mulfactor_dith_simd);
					converted32_hi = _mm_cvtps_epi32(mulled_hi);
				}
				else {
					// new_pixel = (uint8_t)(srcp0[x] + corr);
					converted32_lo = pixel_i_lo;
					converted32_hi = pixel_i_hi;
				}

				__m128i converted_16 = _mm_packs_epi32(converted32_lo, converted32_hi);
        if constexpr(max_pixel_value_dithered <= 16384) // when <= 14 bits. otherwise packus_epi16 handles well. min_epi16 is sse2 only unlike min_epu16
					converted_16 = _mm_min_epi16(converted_16, _mm_set1_epi16(max_pixel_value_dithered)); // new_pixel = min(new_pixel, max_pixel_value_dithered); // clamp upper

																																																// scale back to the required bit depth
																																																// for generality. Now target == 8 bit, and dither_target is also 8 bit
																																																// for test: source:10 bit, target=8 bit, dither_target=4 bit
				const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
        if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0)  // dither to 8, target to 8
					converted_16 = _mm_slli_epi16(converted_16, BITDIFF_BETWEEN_DITHER_AND_TARGET); // new_pixel << BITDIFF_BETWEEN_DITHER_AND_TARGET; // if implemented non-8bit dither target, this should be fullscale

				__m128i converted_8 = _mm_packus_epi16(converted_16, zero);

				_mm_storel_epi64(reinterpret_cast<__m128i *>(&dstp[x]), converted_8);
			}
		} // x
		dstp += dst_pitch;
		srcp0 += src_pitch;
	}
}

// idea borrowed from fmtConv
#define FS_OPTIMIZED_SERPENTINE_COEF

template<int direction>
static __forceinline void diffuse_floyd(int err, int &nextError, int *error_ptr)
{
#if defined (FS_OPTIMIZED_SERPENTINE_COEF)
	const int      e1 = 0;
	const int      e3 = (err * 4 + 8) >> 4;
#else
	const int      e1 = (err + 8) >> 4;
	const int      e3 = (err * 3 + 8) >> 4;
#endif
	const int      e5 = (err * 5 + 8) >> 4;
	const int      e7 = err - e1 - e3 - e5;

	nextError = error_ptr[direction];
	error_ptr[-direction] += e3;
	error_ptr[0] += e5;
	error_ptr[direction] = e1;
	nextError += e7;
}

template<int direction>
static void diffuse_floyd_f(float err, float &nextError, float *error_ptr)
{
#if defined (FS_OPTIMIZED_SERPENTINE_COEF)
	const float    e1 = 0;
	const float    e3 = err * (4.0f / 16);
#else
	const float    e1 = err * (1.0f / 16);
	const float    e3 = err * (3.0f / 16);
#endif
	const float    e5 = err * (5.0f / 16);
	const float    e7 = err * (7.0f / 16);

	nextError = error_ptr[direction];
	error_ptr[-direction] += e3;
	error_ptr[0] += e5;
	error_ptr[direction] = e1;
	nextError += e7;
}

template<typename source_pixel_t, typename target_pixel_t, uint8_t sourcebits, uint8_t TARGET_BITDEPTH, int TARGET_DITHER_BITDEPTH>
static void convert_uint_floyd_c(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const source_pixel_t *srcp = reinterpret_cast<const source_pixel_t *>(srcp8);
	src_pitch = src_pitch / sizeof(source_pixel_t);
	int src_width = src_rowsize / sizeof(source_pixel_t);

	target_pixel_t *dstp = reinterpret_cast<target_pixel_t *>(dstp8);
	dst_pitch = dst_pitch / sizeof(target_pixel_t);

	const int max_pixel_value = (1 << TARGET_BITDEPTH) - 1;
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);

	int *error_ptr_safe = new int[1 + src_width + 1]; // accumulated errors
	std::fill_n(error_ptr_safe, src_width + 2, 0);

	int *error_ptr = error_ptr_safe + 1;

	const int INTERNAL_BITS = DITHER_BIT_DIFF < 6 ? sourcebits + 8 : sourcebits; // keep accuracy
	const int SHIFTBITS_TO_INTERNAL = INTERNAL_BITS - sourcebits;
	const int SHIFTBITS_FROM_INTERNAL = INTERNAL_BITS - TARGET_DITHER_BITDEPTH;
	const int ROUNDER = 1 << (SHIFTBITS_FROM_INTERNAL - 1); // rounding

	for (int y = 0; y < src_height; y++)
	{
		int nextError = error_ptr[0];
		// serpentine forward
		if ((y & 1) == 0)
		{
			for (int x = 0; x < src_width; x++)
			{
				int err = nextError;
				int new_pixel = srcp[x] << SHIFTBITS_TO_INTERNAL; // if necessary
				int sum = new_pixel + err;
				int quantized = (sum + ROUNDER) >> (SHIFTBITS_FROM_INTERNAL);
				err = sum - (quantized << SHIFTBITS_FROM_INTERNAL);
				quantized <<= BITDIFF_BETWEEN_DITHER_AND_TARGET;
				int pix = max(min(max_pixel_value, quantized), 0); // clamp to target bit
				dstp[x] = (target_pixel_t)pix;
				diffuse_floyd<1>(err, nextError, error_ptr + x);
			}
		}
		else {
			// serpentine backward
			for (int x = src_width - 1; x >= 0; --x)
			{
				int err = nextError;
				int new_pixel = srcp[x] << SHIFTBITS_TO_INTERNAL; // if necessary
				int sum = new_pixel + err;
				int quantized = (sum + ROUNDER) >> (SHIFTBITS_FROM_INTERNAL);
				err = sum - (quantized << SHIFTBITS_FROM_INTERNAL);
				quantized <<= BITDIFF_BETWEEN_DITHER_AND_TARGET;
				int pix = max(min(max_pixel_value, quantized), 0); // clamp to target bit
				dstp[x] = (target_pixel_t)pix;
				diffuse_floyd<-1>(err, nextError, error_ptr + x);
			}
		}
		error_ptr[0] = nextError;
		dstp += dst_pitch;
		srcp += src_pitch;
	}

	delete[] error_ptr_safe;
}


// YUV conversions (bit shifts)
// BitDepthConvFuncPtr
// Conversion from 16-14-12-10 to 8 bits (bitshift: 8-6-4-2)
template<uint8_t sourcebits, int dither_mode, int TARGET_DITHER_BITDEPTH>
static void convert_uint16_to_8_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
	src_pitch = src_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);

	int _y = 0; // for ordered dither

	const int TARGET_BITDEPTH = 8; // here is constant (uint8_t target)
	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1;
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but generality)
	const int DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const int DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const int MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data); break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data); break;
	case 6: matrix = reinterpret_cast<const BYTE *>(dither8x8.data); break;
	case 8: matrix = reinterpret_cast<const BYTE *>(dither16x16.data); break;
  default: return; // n/a
	}

	for (int y = 0; y<src_height; y++)
	{
    if constexpr(dither_mode == 0) _y = (y & MASK) << DITHER_ORDER; // ordered dither
		for (int x = 0; x < src_width; x++)
		{
      if constexpr(dither_mode < 0) // -1: no dither
				dstp[x] = srcp0[x] >> (sourcebits - TARGET_BITDEPTH); // no dithering, no range conversion, simply shift
			else { // dither_mode == 0 -> ordered dither
				int corr = matrix[_y | (x & MASK)];
				//BYTE new_pixel = (((srcp0[x] << PRESHIFT) >> (sourcebits - 8)) + corr) >> PRESHIFT; // >> (sourcebits - 8);
				int new_pixel = ((srcp0[x] + corr) >> DITHER_BIT_DIFF);
				new_pixel = min(new_pixel, max_pixel_value_dithered); // clamp upper
																															// scale back to the required bit depth
																															// for generality. Now target == 8 bit, and dither_target is also 8 bit
																															// for test: source:10 bit, target=8 bit, dither_target=4 bit
				const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
        if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0)  // dither to 8, target to 8
					new_pixel = new_pixel << BITDIFF_BETWEEN_DITHER_AND_TARGET; // closest in palette: simple shift with
				dstp[x] = (BYTE)new_pixel;
			}
		}
		dstp += dst_pitch;
		srcp0 += src_pitch;
	}
}

template<uint8_t sourcebits>
static void convert_uint16_to_8_sse2(const BYTE *srcp8, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	src_pitch = src_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);
	int wmod16 = (src_width / 16) * 16;

	__m128i zero = _mm_setzero_si128();
	// no dithering, no range conversion, simply shift
	for (int y = 0; y < src_height; y++)
	{
		for (int x = 0; x < src_width; x += 16)
		{
			__m128i src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 8* uint16
			__m128i src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x + 8));
			src_lo = _mm_srli_epi16(src_lo, (sourcebits - 8));
			src_hi = _mm_srli_epi16(src_hi, (sourcebits - 8));
			__m128i dst = _mm_packus_epi16(src_lo, src_hi);
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), dst);
		}
		// rest
		for (int x = wmod16; x < src_width; x++)
		{
			dstp[x] = srcp[x] >> (sourcebits - 8);
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4305 4309)
#endif

template<uint8_t sourcebits, uint8_t TARGET_DITHER_BITDEPTH>
static void convert_uint16_to_8_dither_sse2(const BYTE *srcp8, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	// Full general ordered dither from 10-16 bits to 2-8 bits, keeping the final 8 bit depth
	// Avisynth's ConvertBits parameter "dither_bits" default 8 goes to TARGET_DITHER_BITDEPTH
	// TARGET_BITDEPTH is always 8 bits, but the dither target can be less than 8.
	// The difference between source bitdepth and TARGET_DITHER_BITDEPTH cannot be more than 8
	// Basic usage: dither down to 8 bits from 10-16 bits.
	// Exotic usage: dither down to 2 bits from 10 bits

	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	src_pitch = src_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t); // real width. We take 2x8 word pixels at a time
	int wmod16 = (src_width / 16) * 16;

	int _y_c = 0; // Bayer matrix shift for ordered dither

	const uint8_t TARGET_BITDEPTH = 8; // here is constant (uint8_t target)
	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1; //may be less than 255, e.g. 15 for dither target 4 bits

	const __m128i max_pixel_value_dithered_epi8 = _mm_set1_epi8(max_pixel_value_dithered);
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const uint8_t DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const uint8_t PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but for the sake of generality)
	const uint8_t DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const uint8_t DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const uint8_t MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	const BYTE *matrix_c;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data_sse2);
		matrix_c = reinterpret_cast<const BYTE *>(dither2x2.data);
		break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data_sse2);
		matrix_c = reinterpret_cast<const BYTE *>(dither4x4.data);
		break;
	case 6:
		matrix = reinterpret_cast<const BYTE *>(dither8x8.data_sse2);
		matrix_c = reinterpret_cast<const BYTE *>(dither8x8.data);
		break;
	case 8:
		matrix = reinterpret_cast<const BYTE *>(dither16x16.data);
		matrix_c = matrix;
		break;
  default: return; // n/a
	}

	const BYTE *current_matrix_line;

	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y < src_height; y++)
	{
		_y_c = (y & MASK) << DITHER_ORDER; // matrix lines stride for C
		current_matrix_line = matrix + ((y & MASK) << 4); // always 16 byte boundary

		__m128i corr = _mm_load_si128(reinterpret_cast<const __m128i*>(current_matrix_line)); // int corr = matrix[_y | (x & MASK)];
		__m128i corr_lo = _mm_unpacklo_epi8(corr, zero); // lower 8 byte->uint16_t
		__m128i corr_hi = _mm_unpackhi_epi8(corr, zero); // upper 8 byte->uint16_t

		for (int x = 0; x < src_width; x += 16)
		{
			__m128i src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 8* uint16
			__m128i src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x + 8));

			// int new_pixel = ((srcp0[x] + corr) >> DITHER_BIT_DIFF);

			__m128i new_pixel_lo, new_pixel_hi;

      if constexpr(sourcebits < 16) { // no overflow
				new_pixel_lo = _mm_srli_epi16(_mm_add_epi16(src_lo, corr_lo), DITHER_BIT_DIFF);
				new_pixel_hi = _mm_srli_epi16(_mm_add_epi16(src_hi, corr_hi), DITHER_BIT_DIFF);
				// scale down after adding dithering noise
			}
			else { // source bits: 16. Overflow can happen when 0xFFFF it dithered up. Go 32 bits
						 // lower
				__m128i src_lo_lo = _mm_unpacklo_epi16(src_lo, zero);
				__m128i corr_lo_lo = _mm_unpacklo_epi16(corr_lo, zero);
				__m128i new_pixel_lo_lo = _mm_srli_epi32(_mm_add_epi32(src_lo_lo, corr_lo_lo), DITHER_BIT_DIFF);

				__m128i src_lo_hi = _mm_unpackhi_epi16(src_lo, zero);
				__m128i corr_lo_hi = _mm_unpackhi_epi16(corr_lo, zero);
				__m128i new_pixel_lo_hi = _mm_srli_epi32(_mm_add_epi32(src_lo_hi, corr_lo_hi), DITHER_BIT_DIFF);

				new_pixel_lo = _mm_packs_epi32(new_pixel_lo_lo, new_pixel_lo_hi); // packs is enough
																																					// upper
				__m128i src_hi_lo = _mm_unpacklo_epi16(src_hi, zero);
				__m128i corr_hi_lo = _mm_unpacklo_epi16(corr_hi, zero);
				__m128i new_pixel_hi_lo = _mm_srli_epi32(_mm_add_epi32(src_hi_lo, corr_hi_lo), DITHER_BIT_DIFF);

				__m128i src_hi_hi = _mm_unpackhi_epi16(src_hi, zero);
				__m128i corr_hi_hi = _mm_unpackhi_epi16(corr_hi, zero);
				__m128i new_pixel_hi_hi = _mm_srli_epi32(_mm_add_epi32(src_hi_hi, corr_hi_hi), DITHER_BIT_DIFF);

				new_pixel_hi = _mm_packs_epi32(new_pixel_hi_lo, new_pixel_hi_hi); // packs is enough
			}

			__m128i new_pixel = _mm_packus_epi16(new_pixel_lo, new_pixel_hi); // 2x8 x16 bit -> 16 byte. Clamp is automatic

      if constexpr(TARGET_DITHER_BITDEPTH < 8) { // generic (not used) fun option to dither 10->4 bits then back to 8 bit
				new_pixel = _mm_min_epu8(new_pixel, max_pixel_value_dithered_epi8);
			}

			const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
      if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0) { //==0 when dither and target are both 8
																										// scale back, when e.g. 10 bit data is dithered down to 4,6,8 bits but the target bit depth is still 8 bit.
				new_pixel = _mm_and_si128(_mm_set1_epi8((0xFF << BITDIFF_BETWEEN_DITHER_AND_TARGET) & 0xFF), _mm_slli_epi32(new_pixel, BITDIFF_BETWEEN_DITHER_AND_TARGET));
				// non-existant _mm_slli_epi8. closest in palette: simple shift
			}

			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), new_pixel);
		}

		// rest, C
		for (int x = wmod16; x < src_width; x++)
		{
			int corr = matrix_c[_y_c | (x & MASK)];
			//BYTE new_pixel = (((srcp0[x] << PRESHIFT) >> (sourcebits - 8)) + corr) >> PRESHIFT; // >> (sourcebits - 8);
			int new_pixel = ((srcp[x] + corr) >> DITHER_BIT_DIFF);
			new_pixel = min(new_pixel, max_pixel_value_dithered); // clamp upper
																														// scale back to the required bit depth
																														// for generality. Now target == 8 bit, and dither_target is also 8 bit
																														// for test: source:10 bit, target=8 bit, dither_target=4 bit
			const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
      if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0)  // dither to 8, target to 8
				new_pixel = new_pixel << BITDIFF_BETWEEN_DITHER_AND_TARGET; // closest in palette: simple shift with
			dstp[x] = (BYTE)new_pixel;
		}

		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

// 10-16bits: sse4.1
// 8 bits: sse2
template<typename pixel_t, uint8_t targetbits, bool chroma, bool fulls, bool fulld>
void convert_32_to_uintN_sse(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const float *srcp = reinterpret_cast<const float *>(srcp8);
	pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);

	src_pitch = src_pitch / sizeof(float);
	dst_pitch = dst_pitch / sizeof(pixel_t);

	int src_width = src_rowsize / sizeof(float);

	const int max_pixel_value = (1 << targetbits) - 1;
	const __m128i max_pixel_value_128 = _mm_set1_epi16(max_pixel_value);

  const int limit_lo_d = (fulld ? 0 : 16) << (targetbits - 8);
  const int limit_hi_d = fulld ? ((1 << targetbits) - 1) : ((chroma ? 240 : 235) << (targetbits - 8));
  const float range_diff_d = (float)limit_hi_d - limit_lo_d;

  const int limit_lo_s = fulls ? 0 : 16;
  const int limit_hi_s = fulls ? 255 : (chroma ? 240 : 235);
  const float range_diff_s = (limit_hi_s - limit_lo_s) / 255.0f;

  // fulls fulld luma             luma_new   chroma                          chroma_new
  // true  false 0..1              16-235     -0.5..0.5                      16-240       Y = Y * ((235-16) << (bpp-8)) + 16, Chroma= Chroma * ((240-16) << (bpp-8)) + 16
  // true  true  0..1               0-255     -0.5..0.5                      0-128-255
  // false false 16/255..235/255   16-235     (16-128)/255..(240-128)/255    16-240
  // false true  16/255..235/255    0..1      (16-128)/255..(240-128)/255    0-128-255
  const float factor = range_diff_d / range_diff_s;

	const float half_i = (float)(1 << (targetbits - 1));
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
	const __m128 half_ps = _mm_set1_ps(0.5f);
#endif
	const __m128 halfint_plus_rounder_ps = _mm_set1_ps(half_i + 0.5f);
  const __m128 limit_lo_s_ps = _mm_set1_ps(limit_lo_s / 255.0f);
  const __m128 limit_lo_plus_rounder_ps = _mm_set1_ps(limit_lo_d + 0.5f);

  __m128 factor_ps = _mm_set1_ps(factor); // 0-1.0 -> 0..max_pixel_value

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x += 8) // 8 pixels at a time
		{
			__m128i result;
			__m128i result_0, result_1;
			__m128 src_0 = _mm_load_ps(reinterpret_cast<const float *>(srcp + x));
			__m128 src_1 = _mm_load_ps(reinterpret_cast<const float *>(srcp + x + 4));
			if (chroma) {
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
				// shift 0.5 before, shift back half_int after. 0.5->exact half of 128/512/...
				src_0 = _mm_sub_ps(src_0, half_ps);
				src_1 = _mm_sub_ps(src_1, half_ps);
				//pixel = (srcp0[x] - 0.5f) * factor + half + 0.5f;
#else
        //pixel = (srcp0[x]       ) * factor + half + 0.5f;
#endif
				src_0 = _mm_add_ps(_mm_mul_ps(src_0, factor_ps), halfint_plus_rounder_ps);
				src_1 = _mm_add_ps(_mm_mul_ps(src_1, factor_ps), halfint_plus_rounder_ps);
			}
			else {
        if constexpr(!fulls) {
          src_0 = _mm_sub_ps(src_0, limit_lo_s_ps);
          src_1 = _mm_sub_ps(src_1, limit_lo_s_ps);
        }
        src_0 = _mm_add_ps(_mm_mul_ps(src_0, factor_ps), limit_lo_plus_rounder_ps);
        src_1 = _mm_add_ps(_mm_mul_ps(src_1, factor_ps), limit_lo_plus_rounder_ps);
        //pixel = (srcp0[x] - limit_lo_s_ps) * factor + half + limit_lo + 0.5f;
			}
			result_0 = _mm_cvttps_epi32(src_0); // truncate
			result_1 = _mm_cvttps_epi32(src_1);
      if constexpr(sizeof(pixel_t) == 2) {
				result = _mm_packus_epi32(result_0, result_1); // sse41
        if constexpr(targetbits > 8 && targetbits < 16) {
					result = _mm_min_epu16(result, max_pixel_value_128); // sse41, extra clamp for 10, 12, 14 bits
				}
				_mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), result);
			}
			else {
				result = _mm_packus_epi32(result_0, result_1);
				result = _mm_packus_epi16(result, result); // lo 8 byte
				_mm_storel_epi64(reinterpret_cast<__m128i *>(dstp + x), result);
			}
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// float to 8 bit, float to 10/12/14/16 bit
template<typename pixel_t, uint8_t targetbits, bool chroma, bool fulls, bool fulld>
static void convert_32_to_uintN_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const float *srcp0 = reinterpret_cast<const float *>(srcp);
	pixel_t *dstp0 = reinterpret_cast<pixel_t *>(dstp);

	src_pitch = src_pitch / sizeof(float);
	dst_pitch = dst_pitch / sizeof(pixel_t);

	int src_width = src_rowsize / sizeof(float);

	const float max_dst_pixelvalue = (float)((1 << targetbits) - 1); // 255, 1023, 4095, 16383, 65535.0
	const float half = (float)(1 << (targetbits - 1));

  const int limit_lo_d = (fulld ? 0 : 16) << (targetbits - 8);
  const int limit_hi_d = fulld ? ((1 << targetbits) - 1) : ((chroma ? 240 : 235) << (targetbits - 8));
  const float range_diff_d = (float)limit_hi_d - limit_lo_d;

  const int limit_lo_s = fulls ? 0 : 16;
  const float limit_lo_s_ps = limit_lo_s / 255.0f;
  const int limit_hi_s = fulls ? 255 : (chroma ? 240 : 235);
  const float range_diff_s = (limit_hi_s - limit_lo_s) / 255.0f; 

  // fulls fulld luma             luma_new   chroma                          chroma_new
  // true  false 0..1              16-235     -0.5..0.5                      16-240       Y = Y * ((235-16) << (bpp-8)) + 16, Chroma= Chroma * ((240-16) << (bpp-8)) + 16
  // true  true  0..1               0-255     -0.5..0.5                      0-128-255
  // false false 16/255..235/255   16-235     (16-128)/255..(240-128)/255    16-240
  // false true  16/255..235/255    0..1      (16-128)/255..(240-128)/255    0-128-255
  const float factor = range_diff_d / range_diff_s;

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			float pixel;
			if (chroma) {
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
				// shift 0.5 before, shift back half_int after. 0.5->exact half of 128/512/...
        pixel = (srcp0[x] - 0.5f);
#else
        pixel = srcp0[x];
#endif
        pixel = pixel * factor + half + 0.5f;
			}
			else {
        if constexpr(!fulls)
          pixel = (srcp0[x] - limit_lo_s_ps) * factor + 0.5f + limit_lo_d;
        else
          pixel = srcp0[x] * factor + 0.5f + limit_lo_d;
			}
			dstp0[x] = pixel_t(clamp(pixel, 0.0f, max_dst_pixelvalue)); // we clamp here!
		}
		dstp0 += dst_pitch;
		srcp0 += src_pitch;
	}
}

// rgb/alpha: full scale. No bit shift, scale full ranges
template<uint8_t targetbits>
static void convert_rgb_8_to_uint16_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint8_t *srcp0 = reinterpret_cast<const uint8_t *>(srcp);
	uint16_t *dstp0 = reinterpret_cast<uint16_t *>(dstp);

	src_pitch = src_pitch / sizeof(uint8_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	int src_width = src_rowsize / sizeof(uint8_t);

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			// test
            if constexpr(targetbits==16)
				dstp0[x] = srcp0[x] * 257; // full range 0..255 <-> 0..65535 (257 = 65535 / 255)
            else if constexpr(targetbits==14)
				dstp0[x] = srcp0[x] * 16383 / 255; // full range 0..255 <-> 0..16384-1
            else if constexpr(targetbits==12)
				dstp0[x] = srcp0[x] * 4095 / 255; // full range 0..255 <-> 0..4096-1
            else if constexpr(targetbits==10)
				dstp0[x] = srcp0[x] * 1023 / 255; // full range 0..255 <-> 0..1024-1
		}
		dstp0 += dst_pitch;
		srcp0 += src_pitch;
	}
}

#if 0
// leave it here, maybe we can use it later
// Tricky simd implementation of integer div 255 w/o division
static inline __m128i Div_4xint32_by_255(const __m128i &esi, const __m128i &magic255div) {
	// simd implementation of
	/*
	Trick of integer/255 w/o division:
	tmp = (int)((( (__int64)esi * (-2139062143)) >> 32) & 0xFFFFFFFF) + esi) >> 7
	result = tmp + (tmp >> 31)

	movzx	eax, BYTE PTR [ecx+edi] // orig pixel
	imul	esi, eax, 16383         // * Scale_Multiplier
	// div 255 follows
	// result in esi is int32
	// Div_4xint32_by_255 implementation from here!
	mov	eax, -2139062143			; 80808081H
	imul	esi  // signed!
	add	edx, esi
	sar	edx, 7
	mov	eax, edx
	shr	eax, 31					; 0000001fH
	add	eax, edx
	mov	WORD PTR [ebx+ecx*2], ax
	*/
	// edx_eax_64 = mulres_lo(esi) * magic255div(eax)
	// _mm_mul_epu32: r64_0 := a0 * b0, r64_1 := a2 * b2 (edx_eax edx_eax)
	// signed mul!
	__m128i mulwithmagic02 = _mm_mul_epi32(esi, magic255div); // signed! need epi not epu! only sse4.1
	__m128i mulwithmagic13 = _mm_mul_epi32(_mm_srli_si128(esi, 4), magic255div);
	// shuffle hi32bit of results to [63..0] and pack. a3->a1, a1->a0
	__m128i upper32bits_edx = _mm_unpacklo_epi32(_mm_shuffle_epi32(mulwithmagic02, _MM_SHUFFLE(0, 0, 3, 1)), _mm_shuffle_epi32(mulwithmagic13, _MM_SHUFFLE(0, 0, 3, 1)));

	// vvv lower 32 bit of result is never used in the algorithm
	// shuffle lo32bit results to [63..0] and pack
	// __m128i lower32bits_eax = _mm_unpacklo_epi32(_mm_shuffle_epi32(mulwithmagic02, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(mulwithmagic13, _MM_SHUFFLE (0,0,2,0)));

	// add edx, mulres_lo(esi)
	__m128i tmp_edx = _mm_add_epi32(upper32bits_edx, esi);
	// sar edx, 7
	// shift arithmetic
	tmp_edx = _mm_srai_epi32(tmp_edx, 7);
	// mov eax, edx
	// shr eax, 31					; 0000001fH
	// shift logical
	__m128i tmp_eax = _mm_srli_epi32(tmp_edx, 31);
	// add eax, edx
	__m128i result = _mm_add_epi32(tmp_eax, tmp_edx);
	return result;
	// 4 results in the lower 16 bits of 4x32 bit register
}
#endif

template<uint8_t targetbits>
static void convert_rgb_8_to_uint16_sse2(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint8_t *srcp = reinterpret_cast<const uint8_t *>(srcp8);
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);

	src_pitch = src_pitch / sizeof(uint8_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	int src_width = src_rowsize / sizeof(uint8_t);
	int wmod16 = (src_width / 16) * 16;

	const int MUL = (targetbits == 16) ? 257 : ((1 << targetbits) - 1);
	const int DIV = (targetbits == 16) ? 1 : 255;
	// 16 bit: one mul only, no need for /255
	// for others: // *16383 *4095 *1023  and /255

	__m128i zero = _mm_setzero_si128();
	__m128i multiplier = _mm_set1_epi16(MUL);
	__m128i magic255div = _mm_set1_epi32(-2139062143); // 80808081H
	__m128 multiplier_float = _mm_set1_ps((float)MUL / DIV);
	// This is ok, since the default SIMD rounding mode is round-to-nearest unlike c++ truncate
	// in C: 1023 * multiplier = 1022.999 -> truncates.

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x += 16)
		{
			__m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 16* uint8
			__m128i src_lo = _mm_unpacklo_epi8(src, zero);             // 8* uint16
			__m128i src_hi = _mm_unpackhi_epi8(src, zero);             // 8* uint16
																																 // test
      if constexpr(targetbits==16) {
				// *257 mullo is faster than x*257 = (x<<8 + x) add/or solution (i7)
				__m128i res_lo = _mm_mullo_epi16(src_lo, multiplier); // lower 16 bit of multiplication is enough
				__m128i res_hi = _mm_mullo_epi16(src_hi, multiplier);
				// dstp[x] = srcp[x] * 257; // RGB: full range 0..255 <-> 0..65535 (257 = 65535 / 255)
				_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), res_lo);
				_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), res_hi);
			}
			else {
#if 0
				if (false) {
					// dead end
					// simulate integer tricky div 255 arithmetic.
					// Unfortunately it's sse41 only plus much slower than float, but still much faster than C. Too much overhead

					// process 8*uint16_t
					//--------------
					// first src_lo

					// imul	esi, eax, 16383
					__m128i res_lower16bit = _mm_mullo_epi16(src_lo, multiplier); // *16383 *4095 *1023 result: int32. get lower 16
					__m128i res_upper16bit = _mm_mulhi_epi16(src_lo, multiplier); // *16383 *4095 *1023 result: int32. get upper 16
					__m128i mulres_lo = _mm_unpacklo_epi16(res_lower16bit, res_upper16bit); // 4 int32
					__m128i mulres_hi = _mm_unpackhi_epi16(res_lower16bit, res_upper16bit); // 4 int32

																																									// process first 4 of 8 uint32_t (mulres_lo)
					__m128i tmp_eax_lo = Div_4xint32_by_255(mulres_lo, magic255div);
					// process second 4 of 8 uint32_t (mulres_hi)
					__m128i tmp_eax_hi = Div_4xint32_by_255(mulres_hi, magic255div);
					__m128i dst = _mm_packus_epi32(tmp_eax_lo, tmp_eax_hi);
					_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), dst);

					//--------------
					// second src_hi
					{
						// imul	esi, eax, 16383|4095|1023
						__m128i res_lower16bit = _mm_mullo_epi16(src_hi, multiplier); // *16383 *4095 *1023 result: int32. get lower 16
						__m128i res_upper16bit = _mm_mulhi_epi16(src_hi, multiplier); // *16383 *4095 *1023 result: int32. get upper 16
						__m128i mulres_lo = _mm_unpacklo_epi16(res_lower16bit, res_upper16bit); // 4 int32
						__m128i mulres_hi = _mm_unpackhi_epi16(res_lower16bit, res_upper16bit); // 4 int32

																																										// process first 4 of 8 uint32_t (mulres_lo)
						__m128i tmp_eax_lo = Div_4xint32_by_255(mulres_lo, magic255div);
						// process second 4 of 8 uint32_t (mulres_hi)
						__m128i tmp_eax_hi = Div_4xint32_by_255(mulres_hi, magic255div);
						__m128i dst = _mm_packus_epi32(tmp_eax_lo, tmp_eax_hi);
						_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), dst);
					}
				}
				else
#endif
				{
					// src_lo: 8*uint16
					// convert to int32 then float, multiply and convert back
					__m128 res_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(src_lo, zero)), multiplier_float);
					__m128 res_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(src_lo, zero)), multiplier_float);
					// Converts the four single-precision, floating-point values of a to signed 32-bit integer values.
					__m128i result_l = _mm_cvtps_epi32(res_lo); // The default rounding mode is round-to-nearest unlike c++ truncate
					__m128i result_h = _mm_cvtps_epi32(res_hi);
					// Pack and store no need for packus for <= 14 bit
					__m128i result = _mm_packs_epi32(result_l, result_h); // 4*32+4*32 = 8*16
					_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);

					// src_hi: 8*uint16
					// convert to int32 then float, multiply and convert back
					res_lo = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(src_hi, zero)), multiplier_float);
					res_hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(src_hi, zero)), multiplier_float);
					// Converts the four single-precision, floating-point values of a to signed 32-bit integer values.
					result_l = _mm_cvtps_epi32(res_lo);
					result_h = _mm_cvtps_epi32(res_hi);
					// Pack and store no need for packus for <= 14 bit
					result = _mm_packs_epi32(result_l, result_h); // 4*32+4*32 = 8*16
					_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), result);
				}
			} // if 16 bit else
		} // for x
			// rest
		for (int x = wmod16; x < src_width; x++)
		{
			dstp[x] = srcp[x] * MUL / DIV; // RGB: full range 0..255 <-> 0..16384-1
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	} // for y
}


// YUV: bit shift 8 to 10-12-14-16 bits
template<uint8_t targetbits>
static void convert_8_to_uint16_c(const BYTE *srcp, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);

	dst_pitch = dst_pitch / sizeof(uint16_t);

	int src_width = src_rowsize / sizeof(uint8_t); // intentional

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			dstp[x] = srcp[x] << (targetbits - 8);
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

template<uint8_t targetbits>
static void convert_8_to_uint16_sse2(const BYTE *srcp, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);

	dst_pitch = dst_pitch / sizeof(uint16_t);

	int src_width = src_rowsize / sizeof(uint8_t);
	int wmod16 = (src_width / 16) * 16;

	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x += 16)
		{
			__m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 16 bytes
			__m128i dst_lo = _mm_unpacklo_epi8(src, zero);
			__m128i dst_hi = _mm_unpackhi_epi8(src, zero);
			dst_lo = _mm_slli_epi16(dst_lo, (targetbits - 8));
			dst_hi = _mm_slli_epi16(dst_hi, (targetbits - 8));
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), dst_lo);
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), dst_hi);
		}
		// rest
		for (int x = wmod16; x < src_width; x++)
		{
			dstp[x] = srcp[x] << (targetbits - 8);
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

// RGB full range: 10-12-14-16 <=> 10-12-14-16 bits
template<uint8_t sourcebits, uint8_t targetbits, bool hasSSE4>
static void convert_rgb_uint16_to_uint16_sse2(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	src_pitch = src_pitch / sizeof(uint16_t);
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);
	dst_pitch = dst_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);
	int wmod = (src_width / 8) * 8;

	const uint16_t source_max = (1 << sourcebits) - 1;
	const uint16_t target_max = (1 << targetbits) - 1;

	__m128 factor = _mm_set1_ps((float)target_max / source_max);
	__m128i max_pixel_value = _mm_set1_epi16(target_max);
	__m128i zero = _mm_setzero_si128();

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x += 8)
		{
			__m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 8* uint16

			__m128i src_lo = _mm_unpacklo_epi16(src, zero);
			__m128i src_hi = _mm_unpackhi_epi16(src, zero);

			__m128 result_lo = _mm_mul_ps(_mm_cvtepi32_ps(src_lo), factor);
			__m128 result_hi = _mm_mul_ps(_mm_cvtepi32_ps(src_hi), factor);

			__m128i result;
			if (hasSSE4)
				result = _mm_packus_epi32(_mm_cvtps_epi32(result_lo), _mm_cvtps_epi32(result_hi));
			else
				result = _MM_PACKUS_EPI32(_mm_cvtps_epi32(result_lo), _mm_cvtps_epi32(result_hi));
      if constexpr(targetbits < 16) {
				if (hasSSE4)
					result = _mm_min_epu16(result, max_pixel_value);
				else
					result = _MM_MIN_EPU16(result, max_pixel_value);
			}
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
		}
		// rest
		for (int x = wmod; x < src_width; x++)
		{
			dstp[x] = (uint16_t)((int64_t)srcp[x] * target_max / source_max); // expand range
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}


// RGB full range: 10-12-14-16 <=> 10-12-14-16 bits
template<uint8_t sourcebits, uint8_t targetbits>
static void convert_rgb_uint16_to_uint16_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
	uint16_t *dstp0 = reinterpret_cast<uint16_t *>(dstp);

	src_pitch = src_pitch / sizeof(uint16_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	const int src_width = src_rowsize / sizeof(uint16_t);

	const uint16_t source_max = (1 << sourcebits) - 1;
	const uint16_t target_max = (1 << targetbits) - 1;

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			// int64: avoid unsigned * unsigned = signed arithmetic overflow
			dstp0[x] = (uint16_t)((int64_t)srcp0[x] * target_max / source_max);
		}
		dstp0 += dst_pitch;
		srcp0 += src_pitch;
	}
}

template<uint8_t sourcebits, uint8_t targetbits, int TARGET_DITHER_BITDEPTH>
static void convert_rgb_uint16_to_uint16_dither_c(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);

	src_pitch = src_pitch / sizeof(uint16_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	const int src_width = src_rowsize / sizeof(uint16_t);

	const int source_max = (1 << sourcebits) - 1;

	int _y = 0; // for ordered dither

	const int TARGET_BITDEPTH = targetbits;
	const int max_pixel_value = (1 << TARGET_BITDEPTH) - 1;
	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1;
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but generality)
	const int DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const int DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const int MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data); break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data); break;
	case 6: matrix = reinterpret_cast<const BYTE *>(dither8x8.data); break;
	case 8: matrix = reinterpret_cast<const BYTE *>(dither16x16.data); break;
  default: return; // n/a
	}

	for (int y = 0; y < src_height; y++)
	{
		_y = (y & MASK) << DITHER_ORDER; // ordered dither
		for (int x = 0; x < src_width; x++)
		{
			int corr = matrix[_y | (x & MASK)];
			//BYTE new_pixel = (((srcp0[x] << PRESHIFT) >> (sourcebits - 8)) + corr) >> PRESHIFT; // >> (sourcebits - 8);
			//int new_pixel = ((srcp[x] + corr) >> DITHER_BIT_DIFF);
			int64_t new_pixel = (int64_t)(srcp[x] + corr) * max_pixel_value_dithered / source_max;

			// new_pixel = min(new_pixel, max_pixel_value_dithered_i); // clamp upper
			// scale back to the required bit depth
			const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
      if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0) {
				new_pixel = new_pixel * max_pixel_value / max_pixel_value_dithered;
			}
			dstp[x] = (uint16_t)(min((int)new_pixel, max_pixel_value));
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}

}

// YUV: bit shift 10-12-14-16 <=> 10-12-14-16 bits
// shift right or left, depending on expandrange template param
template<bool expandrange, uint8_t shiftbits>
static void convert_uint16_to_uint16_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
	uint16_t *dstp0 = reinterpret_cast<uint16_t *>(dstp);

	src_pitch = src_pitch / sizeof(uint16_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	const int src_width = src_rowsize / sizeof(uint16_t);

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			if (expandrange)
				dstp0[x] = srcp0[x] << shiftbits;  // expand range. No clamp before, source is assumed to have valid range
			else
				dstp0[x] = srcp0[x] >> shiftbits;  // reduce range
		}
		dstp0 += dst_pitch;
		srcp0 += src_pitch;
	}
}

// YUV: bit shift 10-12-14-16 <=> 10-12-14-16 bits
// shift right or left, depending on expandrange template param
template<uint8_t sourcebits, uint8_t targetbits, int TARGET_DITHER_BITDEPTH>
static void convert_uint16_to_uint16_dither_c(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);

	src_pitch = src_pitch / sizeof(uint16_t);
	dst_pitch = dst_pitch / sizeof(uint16_t);

	const int src_width = src_rowsize / sizeof(uint16_t);

	int _y = 0; // for ordered dither

	const int TARGET_BITDEPTH = targetbits;
	const int max_pixel_value_dithered = (1 << TARGET_DITHER_BITDEPTH) - 1;
	// precheck ensures:
	// TARGET_BITDEPTH >= TARGET_DITHER_BITDEPTH
	// sourcebits - TARGET_DITHER_BITDEPTH <= 8
	// sourcebits - TARGET_DITHER_BITDEPTH is even (later we can use PRESHIFT)
	const int DITHER_BIT_DIFF = (sourcebits - TARGET_DITHER_BITDEPTH); // 2, 4, 6, 8
	const int PRESHIFT = DITHER_BIT_DIFF & 1;  // 0 or 1: correction for odd bit differences (not used here but generality)
	const int DITHER_ORDER = (DITHER_BIT_DIFF + PRESHIFT) / 2;
	const int DITHER_SIZE = 1 << DITHER_ORDER; // 9,10=2  11,12=4  13,14=8  15,16=16
	const int MASK = DITHER_SIZE - 1;
	// 10->8: 0x01 (2x2)
	// 11->8: 0x03 (4x4)
	// 12->8: 0x03 (4x4)
	// 14->8: 0x07 (8x8)
	// 16->8: 0x0F (16x16)
	const BYTE *matrix;
	switch (sourcebits - TARGET_DITHER_BITDEPTH) {
	case 2: matrix = reinterpret_cast<const BYTE *>(dither2x2.data); break;
	case 4: matrix = reinterpret_cast<const BYTE *>(dither4x4.data); break;
	case 6: matrix = reinterpret_cast<const BYTE *>(dither8x8.data); break;
	case 8: matrix = reinterpret_cast<const BYTE *>(dither16x16.data); break;
  default: return; // n/a
	}

	for (int y = 0; y < src_height; y++)
	{
		_y = (y & MASK) << DITHER_ORDER; // ordered dither
		for (int x = 0; x < src_width; x++)
		{
			int corr = matrix[_y | (x & MASK)];
			//BYTE new_pixel = (((srcp0[x] << PRESHIFT) >> (sourcebits - 8)) + corr) >> PRESHIFT; // >> (sourcebits - 8);
			int new_pixel = ((srcp[x] + corr) >> DITHER_BIT_DIFF);
			new_pixel = min(new_pixel, max_pixel_value_dithered); // clamp upper
																														// scale back to the required bit depth
																														// for generality. Now target == 8 bit, and dither_target is also 8 bit
																														// for test: source:10 bit, target=8 bit, dither_target=4 bit
			const int BITDIFF_BETWEEN_DITHER_AND_TARGET = DITHER_BIT_DIFF - (sourcebits - TARGET_BITDEPTH);
      if constexpr(BITDIFF_BETWEEN_DITHER_AND_TARGET != 0)  // dither to 8, target to 8
				new_pixel = new_pixel << BITDIFF_BETWEEN_DITHER_AND_TARGET; // closest in palette: simple shift with
			dstp[x] = (uint16_t)new_pixel;
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

template<bool expandrange, uint8_t shiftbits>
static void convert_uint16_to_uint16_sse2(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	// remark: Compiler with SSE2 option generates the same effective code like this in C
	// Drawback of SSE2: a future avx2 target gives more efficient code than inline SSE2 (256 bit registers)
	const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
	src_pitch = src_pitch / sizeof(uint16_t);
	uint16_t *dstp = reinterpret_cast<uint16_t *>(dstp8);
	dst_pitch = dst_pitch / sizeof(uint16_t);
	int src_width = src_rowsize / sizeof(uint16_t);
	int wmod = (src_width / 16) * 16;

	__m128i shift = _mm_set_epi32(0, 0, 0, shiftbits);

	// no dithering, no range conversion, simply shift
	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x += 16)
		{
			__m128i src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 8* uint16
			__m128i src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x + 8)); // 8* uint16
			if (expandrange) {
				src_lo = _mm_sll_epi16(src_lo, shift);
				src_hi = _mm_sll_epi16(src_hi, shift);
			}
			else {
				src_lo = _mm_srl_epi16(src_lo, shift);
				src_hi = _mm_srl_epi16(src_hi, shift);
			}
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), src_lo);
			_mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), src_hi);
		}
		// rest
		for (int x = wmod; x < src_width; x++)
		{
			if (expandrange)
				dstp[x] = srcp[x] << shiftbits;  // expand range. No clamp before, source is assumed to have valid range
			else
				dstp[x] = srcp[x] >> shiftbits;  // reduce range
		}
		dstp += dst_pitch;
		srcp += src_pitch;
	}
}

// 8 bit to float, 16/14/12/10 bits to float
template<typename pixel_t, uint8_t sourcebits, bool chroma, bool fulls, bool fulld>
static void convert_uintN_to_float_c(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
	const pixel_t *srcp0 = reinterpret_cast<const pixel_t *>(srcp);
	float *dstp0 = reinterpret_cast<float *>(dstp);

	src_pitch = src_pitch / sizeof(pixel_t);
	dst_pitch = dst_pitch / sizeof(float);

	int src_width = src_rowsize / sizeof(pixel_t);

  const int limit_lo_s = (fulls ? 0 : 16) << (sourcebits - 8);
  const int limit_hi_s = fulls ? ((1 << sourcebits) - 1) : ((chroma ? 240 : 235) << (sourcebits - 8));
  const float range_diff_s = (float)limit_hi_s - limit_lo_s;

  const int limit_lo_d = fulld ? 0 : 16;
  const int limit_hi_d = fulld ? 255 : (chroma ? 240 : 235);
  const float range_diff_d = (limit_hi_d - limit_lo_d) / 255.0f;

  // fulls fulld luma             luma_new   chroma                          chroma_new
  // true  false 0..1              16-235     -0.5..0.5                      16-240       Y = Y * ((235-16) << (bpp-8)) + 16, Chroma= Chroma * ((240-16) << (bpp-8)) + 16
  // true  true  0..1               0-255     -0.5..0.5                      0-128-255
  // false false 16/255..235/255   16-235     (16-128)/255..(240-128)/255    16-240
  // false true  16/255..235/255    0..1      (16-128)/255..(240-128)/255    0-128-255
  const float factor = range_diff_d / range_diff_s;

  const int half = 1 << (sourcebits - 1);

  // 0..255,65535 -> 0..1.0 (or -0.5..+0.5) or less if !full

	for (int y = 0; y<src_height; y++)
	{
		for (int x = 0; x < src_width; x++)
		{
			float pixel;
			if (chroma) {
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
        if (fulls)
          pixel = srcp0[x] * factor; // 0..255->0..1
        else
          pixel = (srcp0[x] - half) * factor + 0.5f; // back to 0..1.0 (0.5 centered)
#else
        if (fulls)
          pixel = srcp0[x] * factor - 0.5f; // 0..1->-0.5..0.5
        else
          pixel = (srcp0[x] - half) * factor; // -0.5..0.5 when fulld
#endif
			}
			else {
        pixel = (srcp0[x] - limit_lo_s) * factor + limit_lo_d / 255.0f;
			}
			dstp0[x] = pixel;
		}
		dstp0 += dst_pitch;
		srcp0 += src_pitch;
	}
	// seems we better stuck with C in the future on such a simple loops
	// if we could put it in a separate file
	// VS2015 AVX2 code for this:
	// takes (8 uint16_t -> 8*float(256 bit) at a time) * unroll_by_2
	// then makes singles with unrolled_by_4 until it can, then do the rest.
	/*
	AVX2 by VS2015: (8*uint16->8*float)xUnrollBy2
	$LL7@convert_ui:
	vpmovzxwd ymm0, XMMWORD PTR [esi+ecx*2]
	vcvtdq2ps ymm0, ymm0
	vmulps	ymm0, ymm0, ymm2
	vmovups	YMMWORD PTR [edi+ecx*4], ymm0
	vpmovzxwd ymm0, XMMWORD PTR [esi+ecx*2+16]
	vcvtdq2ps ymm0, ymm0
	vmulps	ymm0, ymm0, ymm2
	vmovups	YMMWORD PTR [edi+ecx*4+32], ymm0
	add	ecx, 16					; 00000010H
	cmp	ecx, ebx
	jl	SHORT $LL7@convert_ui

	SSE2 by VS2015 (4*uint16->4*float)xUnrollBy2
	$LL7@convert_ui:
	movq	xmm1, QWORD PTR [ebp+ecx*2]
	xorps	xmm0, xmm0
	punpcklwd xmm1, xmm0
	cvtdq2ps xmm0, xmm1
	mulps	xmm0, xmm3
	movups	XMMWORD PTR [ebx+ecx*4], xmm0
	movq	xmm1, QWORD PTR [ebp+ecx*2+8]
	xorps	xmm0, xmm0
	punpcklwd xmm1, xmm0
	cvtdq2ps xmm0, xmm1
	mulps	xmm0, xmm3
	movups	XMMWORD PTR [ebx+ecx*4+16], xmm0
	add	ecx, 8
	cmp	ecx, esi
	jl	SHORT $LL7@convert_ui
	*/
}

BitDepthConvFuncPtr get_convert_to_8_function(bool full_scale, int source_bitdepth, int dither_mode, int dither_bitdepth, int rgb_step, int cpu)
{
	std::map<std::tuple<bool, int, int, int, int, int>, BitDepthConvFuncPtr> func_copy;
	using std::make_tuple;
	/*
	conv_function_full_scale = (sse2 && dither_mode<0) ? convert_rgb_uint16_to_8_c<10, -1, 8, 1> : (dither_mode>=0 ? convert_rgb_uint16_to_8_c<10, 0, 8, 1> : convert_rgb_uint16_to_8_c<10, -1, 8, 1>);
	conv_function_full_scale_no_dither = sse2 ? convert_rgb_uint16_to_8_c<10, -1, 8, 1> : convert_rgb_uint16_to_8_c<10, -1, 8, 1>;
	conv_function_shifted_scale = (sse2 && dither_mode<0) ? convert_uint16_to_8_sse2<10> : (dither_mode>=0 ? convert_uint16_to_8_c<10, 0, 8> : convert_uint16_to_8_c<10, -1, 8>);
	*/
	const int DITHER_TARGET_BITDEPTH_8 = 8;
	const int DITHER_TARGET_BITDEPTH_7 = 7;
	const int DITHER_TARGET_BITDEPTH_6 = 6;
	const int DITHER_TARGET_BITDEPTH_5 = 5;
	const int DITHER_TARGET_BITDEPTH_4 = 4;
	const int DITHER_TARGET_BITDEPTH_3 = 3;
	const int DITHER_TARGET_BITDEPTH_2 = 2;
	const int DITHER_TARGET_BITDEPTH_1 = 1;
	const int DITHER_TARGET_BITDEPTH_0 = 0;

	if (dither_mode < 0)
		dither_bitdepth = 8; // default entry in the tables below
	if (dither_mode == 1) // no special version for fullscale dithering down.
		full_scale = false;

	// full scale

	// no dither, C
	func_copy[make_tuple(true, 10, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<10, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 12, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<12, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 14, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<14, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<16, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	// for RGB48 and RGB64 source
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 3, 0)] = convert_rgb_uint16_to_8_c<16, -1, DITHER_TARGET_BITDEPTH_8, 1>; // dither rgb_step param is n/a
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 4, 0)] = convert_rgb_uint16_to_8_c<16, -1, DITHER_TARGET_BITDEPTH_8, 1>; // dither rgb_step param is n/a

																																																																				//-----------
																																																																				// full scale, no dither, SSE2
	func_copy[make_tuple(true, 10, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<10, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 12, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<12, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 14, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<14, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, -1, DITHER_TARGET_BITDEPTH_8, 1>;
	// for RGB48 and RGB64 source
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 3, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, -1, DITHER_TARGET_BITDEPTH_8, 1>; // dither rgb_step param is n/a
	func_copy[make_tuple(true, 16, -1, DITHER_TARGET_BITDEPTH_8, 4, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, -1, DITHER_TARGET_BITDEPTH_8, 1>; // dither rgb_step param is n/a

																																																																									 //-----------
																																																																									 // full scale, dither, C
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_6, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_rgb_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_4, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_rgb_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_2, 1>;

	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_6, 1>;
	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_rgb_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_4, 1>;

	func_copy[make_tuple(true, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<14, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_8_c<14, 0, DITHER_TARGET_BITDEPTH_6, 1>;

	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_8_c<16, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	// for RGB48 and RGB64 source
	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 3, 0)] = convert_rgb_uint16_to_8_c<16, 0, DITHER_TARGET_BITDEPTH_8, 3>; // dither rgb_step param is filled
	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 4, 0)] = convert_rgb_uint16_to_8_c<16, 0, DITHER_TARGET_BITDEPTH_8, 4>; // dither rgb_step param is filled

																																																																			//-----------
																																																																			// full scale, dither, SSE2
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<10, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<10, 0, DITHER_TARGET_BITDEPTH_6, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<10, 0, DITHER_TARGET_BITDEPTH_4, 1>;
	func_copy[make_tuple(true, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<10, 0, DITHER_TARGET_BITDEPTH_2, 1>;

	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<12, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<12, 0, DITHER_TARGET_BITDEPTH_6, 1>;
	func_copy[make_tuple(true, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<12, 0, DITHER_TARGET_BITDEPTH_4, 1>;

	func_copy[make_tuple(true, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<14, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	func_copy[make_tuple(true, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<14, 0, DITHER_TARGET_BITDEPTH_6, 1>;

	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, 0, DITHER_TARGET_BITDEPTH_8, 1>;
	// for RGB48 and RGB64 source
	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 3, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, 0, DITHER_TARGET_BITDEPTH_8, 3>; // dither rgb_step param is filled
	func_copy[make_tuple(true, 16, 0, DITHER_TARGET_BITDEPTH_8, 4, CPUF_SSE2)] = convert_rgb_uint16_to_8_sse2<16, 0, DITHER_TARGET_BITDEPTH_8, 4>; // dither rgb_step param is filled

																																																																								 //-----------
																																																																								 // Floyd dither, C, dither to 8 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_8>;
	// Floyd dither, C, dither to 7 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_7>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_7>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_7>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_7>;
	// Floyd dither, C, dither to 6 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_6>;
	// Floyd dither, C, dither to 5 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_5>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_5>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_5>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_5>;
	// Floyd dither, C, dither to 4 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_4>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_4>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_4>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_4>;
	// Floyd dither, C, dither to 3 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_3>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_3>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_3>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_3>;
	// Floyd dither, C, dither to 2 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_2>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_2>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_2>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_2>;
	// Floyd dither, C, dither to 1 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_1>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_1>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_1>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_1>;
	// Floyd dither, C, dither to 0 bits
	func_copy[make_tuple(false, 10, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 10, 8, DITHER_TARGET_BITDEPTH_0>;
	func_copy[make_tuple(false, 12, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 12, 8, DITHER_TARGET_BITDEPTH_0>;
	func_copy[make_tuple(false, 14, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 14, 8, DITHER_TARGET_BITDEPTH_0>;
	func_copy[make_tuple(false, 16, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint8_t, 16, 8, DITHER_TARGET_BITDEPTH_0>;

	// shifted scale (YUV)

	// no dither, C
	func_copy[make_tuple(false, 10, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<10, -1, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 12, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<12, -1, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 14, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<14, -1, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 16, -1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<16, -1, DITHER_TARGET_BITDEPTH_8>;
	// no dither, SSE2
	func_copy[make_tuple(false, 10, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_sse2<10>;
	func_copy[make_tuple(false, 12, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_sse2<12>;
	func_copy[make_tuple(false, 14, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_sse2<14>;
	func_copy[make_tuple(false, 16, -1, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_sse2<16>;

	// dither, C, dither to 8 bits
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<14, 0, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_8_c<16, 0, DITHER_TARGET_BITDEPTH_8>;
	// dither, SSE2
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<10, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<12, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<14, DITHER_TARGET_BITDEPTH_8>;
	func_copy[make_tuple(false, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<16, DITHER_TARGET_BITDEPTH_8>;

	// dither, C, dither to 6 bits, max diff 8, allowed from 10-14 bits
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_8_c<14, 0, DITHER_TARGET_BITDEPTH_6>;
	// dither, SSE2
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<10, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<12, DITHER_TARGET_BITDEPTH_6>;
	func_copy[make_tuple(false, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<14, DITHER_TARGET_BITDEPTH_6>;

	// dither, C, dither to 4 bits, max diff 8, allowed from 10-12 bits
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_4>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint16_to_8_c<12, 0, DITHER_TARGET_BITDEPTH_4>;
	// dither, SSE2
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<10, DITHER_TARGET_BITDEPTH_4>;
	func_copy[make_tuple(false, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<12, DITHER_TARGET_BITDEPTH_4>;

	// dither, C, dither to 2 bits, max diff 8, allowed from 10 bits
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint16_to_8_c<10, 0, DITHER_TARGET_BITDEPTH_2>;
	// dither, SSE2
	func_copy[make_tuple(false, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, CPUF_SSE2)] = convert_uint16_to_8_dither_sse2<10, DITHER_TARGET_BITDEPTH_2>;

	BitDepthConvFuncPtr result = func_copy[make_tuple(full_scale, source_bitdepth, dither_mode, dither_bitdepth, rgb_step, cpu)];
	if (result == nullptr)
		result = func_copy[make_tuple(full_scale, source_bitdepth, dither_mode, dither_bitdepth, rgb_step, 0)]; // fallback to C
	return result;
}

BitDepthConvFuncPtr get_convert_to_16_16_down_dither_function(bool full_scale, int source_bitdepth, int target_bitdepth, int dither_mode, int dither_bitdepth, int rgb_step, int cpu)
{
	std::map<std::tuple<bool, int /*src*/, int /*target*/, int /*dithermode*/, int /*ditherbits*/, int /*rgbstep*/, int /*cpu*/>, BitDepthConvFuncPtr> func_copy;
	using std::make_tuple;
	/*
	conv_function_full_scale = (sse2 && dither_mode<0) ? convert_rgb_uint16_to_8_c<10, -1, 8, 1> : (dither_mode>=0 ? convert_rgb_uint16_to_8_c<10, 0, 8, 1> : convert_rgb_uint16_to_8_c<10, -1, 8, 1>);
	conv_function_full_scale_no_dither = sse2 ? convert_rgb_uint16_to_8_c<10, -1, 8, 1> : convert_rgb_uint16_to_8_c<10, -1, 8, 1>;
	conv_function_shifted_scale = (sse2 && dither_mode<0) ? convert_uint16_to_8_sse2<10> : (dither_mode>=0 ? convert_uint16_to_8_c<10, 0, 8> : convert_uint16_to_8_c<10, -1, 8>);
	*/
	const int DITHER_TARGET_BITDEPTH_14 = 14;
	const int DITHER_TARGET_BITDEPTH_12 = 12;
	const int DITHER_TARGET_BITDEPTH_10 = 10;
	const int DITHER_TARGET_BITDEPTH_8 = 8;
	const int DITHER_TARGET_BITDEPTH_7 = 7;
	const int DITHER_TARGET_BITDEPTH_6 = 6;
	const int DITHER_TARGET_BITDEPTH_5 = 5;
	const int DITHER_TARGET_BITDEPTH_4 = 4;
	const int DITHER_TARGET_BITDEPTH_3 = 3;
	const int DITHER_TARGET_BITDEPTH_2 = 2; // only for 10->10 bits, but dithering_bits==2
	const int DITHER_TARGET_BITDEPTH_1 = 1; // FloydSteinberg allows any difference in the implementation
	const int DITHER_TARGET_BITDEPTH_0 = 0; // FloydSteinberg allows any difference in the implementation

	if (dither_mode == 1) // no special version for fullscale dithering down.
		full_scale = false;

	if (full_scale) {
		// 16->10,12,14
		// dither, C, dither to N bits
		func_copy[make_tuple(true, 16, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 16, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 10, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(true, 16, 12, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(true, 16, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 16, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(true, 16, 14, 0, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 14, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(true, 16, 14, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(true, 16, 14, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 16, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(true, 16, 16, 0, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(true, 16, 16, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(true, 16, 16, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 16, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_8>;

		// 14->10,12
		// dither, C, dither to N bits
		func_copy[make_tuple(true, 14, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 14, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 14, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_6>;

		func_copy[make_tuple(true, 14, 12, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(true, 14, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 14, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 14, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_6>;

		func_copy[make_tuple(true, 14, 14, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(true, 14, 14, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 14, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 14, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_6>;

		// 12->10
		// dither, C, dither to N bits
		func_copy[make_tuple(true, 12, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 12, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 12, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(true, 12, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_4>;

		func_copy[make_tuple(true, 12, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(true, 12, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 12, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(true, 12, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_4>;

		// 10->10
		// dither, C, dither to N bits
		func_copy[make_tuple(true, 10, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(true, 10, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(true, 10, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(true, 10, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_rgb_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_2>;
	}
	else {

		// floyd 16->
		// 16->10,12,14
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 16, 10, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 10, DITHER_TARGET_BITDEPTH_0>;

		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 16, 12, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 12, DITHER_TARGET_BITDEPTH_0>;

		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 16, 14, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 14, DITHER_TARGET_BITDEPTH_0>;
		// keeping bit depth but dither down
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 16, 16, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 16, 16, DITHER_TARGET_BITDEPTH_0>;
		// floyd 14->
		// 14->10,12
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 14, 10, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 10, DITHER_TARGET_BITDEPTH_0>;

		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 14, 12, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 12, DITHER_TARGET_BITDEPTH_0>;
		// keeping bit depth but dither down
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 14, 14, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 14, 14, DITHER_TARGET_BITDEPTH_0>;
		// floyd 12->
		// 12->10
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 12, 10, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 10, DITHER_TARGET_BITDEPTH_0>;
		// keeping bit depth but dither down
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 12, 12, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 12, 12, DITHER_TARGET_BITDEPTH_0>;
		// floyd 12->
		// 10->10
		// dither, C, dither to N bits
		// keeping bit depth but dither down
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_7, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_7>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_5, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_5>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_3, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_3>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_2>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_1, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_1>;
		func_copy[make_tuple(false, 10, 10, 1, DITHER_TARGET_BITDEPTH_0, 1, 0)] = convert_uint_floyd_c<uint16_t, uint16_t, 10, 10, DITHER_TARGET_BITDEPTH_0>;

		// end of floyd

		// shifted scale
		// 16->10,12,14
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 16, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 10, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(false, 16, 12, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(false, 16, 14, 0, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 14, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(false, 16, 14, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 14, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 12, DITHER_TARGET_BITDEPTH_8>;

		func_copy[make_tuple(false, 16, 16, 0, DITHER_TARGET_BITDEPTH_14, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_14>;
		func_copy[make_tuple(false, 16, 16, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 16, 16, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 16, 16, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<16, 16, DITHER_TARGET_BITDEPTH_8>;

		// 14->10,12
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 14, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 10, DITHER_TARGET_BITDEPTH_6>;

		func_copy[make_tuple(false, 14, 12, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 14, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 12, DITHER_TARGET_BITDEPTH_6>;

		func_copy[make_tuple(false, 14, 14, 0, DITHER_TARGET_BITDEPTH_12, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_12>;
		func_copy[make_tuple(false, 14, 14, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 14, 14, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 14, 14, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<14, 14, DITHER_TARGET_BITDEPTH_6>;

		// 12->10
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 12, 10, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 12, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 12, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 12, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 10, DITHER_TARGET_BITDEPTH_4>;

		func_copy[make_tuple(false, 12, 12, 0, DITHER_TARGET_BITDEPTH_10, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_10>;
		func_copy[make_tuple(false, 12, 12, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 12, 12, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 12, 12, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint16_to_uint16_dither_c<12, 12, DITHER_TARGET_BITDEPTH_4>;

		// 10->10 only dither down
		// dither, C, dither to N bits
		func_copy[make_tuple(false, 10, 10, 0, DITHER_TARGET_BITDEPTH_8, 1, 0)] = convert_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_8>;
		func_copy[make_tuple(false, 10, 10, 0, DITHER_TARGET_BITDEPTH_6, 1, 0)] = convert_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_6>;
		func_copy[make_tuple(false, 10, 10, 0, DITHER_TARGET_BITDEPTH_4, 1, 0)] = convert_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_4>;
		func_copy[make_tuple(false, 10, 10, 0, DITHER_TARGET_BITDEPTH_2, 1, 0)] = convert_uint16_to_uint16_dither_c<10, 10, DITHER_TARGET_BITDEPTH_2>;

	}
	BitDepthConvFuncPtr result = func_copy[make_tuple(full_scale, source_bitdepth, target_bitdepth, dither_mode, dither_bitdepth, rgb_step, cpu)];
	if (result == nullptr)
		result = func_copy[make_tuple(full_scale, source_bitdepth, target_bitdepth, dither_mode, dither_bitdepth, rgb_step, 0)]; // fallback to C
	return result;
}

template <int SRC_BITS, int TGT_BITS, bool DITHER, bool CHROMA>
BitDepthConvFuncPtr2 get_cuda_conv_bits()
{
	return ConvertBitsKernel<
		(SRC_BITS == TGT_BITS) ? 0 :
		(SRC_BITS == 32) ? 4 :
		(TGT_BITS == 32) ? 5 :
		(SRC_BITS < TGT_BITS) ? 3 :
		(DITHER) ? 1 : 2>::template launch<SRC_BITS, TGT_BITS, DITHER, CHROMA>;
}

template <int SRC_BITS, int TGT_BITS, bool DITHER>
BitDepthConvFuncPtr2 get_cuda_conv_bits(bool chroma)
{
	if (chroma) {
		return get_cuda_conv_bits<SRC_BITS, TGT_BITS, DITHER, true>();
	}
	return get_cuda_conv_bits<SRC_BITS, TGT_BITS, DITHER, false>();
}

template <int SRC_BITS, int TGT_BITS> 
BitDepthConvFuncPtr2 get_cuda_conv_bits(bool dither, bool chroma)
{
	if (dither) {
		return get_cuda_conv_bits<SRC_BITS, TGT_BITS, true>(chroma);
	}
	return get_cuda_conv_bits<SRC_BITS, TGT_BITS, false>(chroma);
}

template <int SRC_BITS>
BitDepthConvFuncPtr2 get_cuda_conv_bits(int tgt_bits, bool dither, bool chroma, PNeoEnv env)
{
	switch (tgt_bits) {
	case 8:
		return get_cuda_conv_bits<SRC_BITS, 8>(dither, chroma);
	case 10:
		return get_cuda_conv_bits<SRC_BITS, 10>(dither, chroma);
	case 12:
		return get_cuda_conv_bits<SRC_BITS, 12>(dither, chroma);
	case 14:
		return get_cuda_conv_bits<SRC_BITS, 14>(dither, chroma);
	case 16:
		return get_cuda_conv_bits<SRC_BITS, 16>(dither, chroma);
	case 32:
		return get_cuda_conv_bits<SRC_BITS, 32>(dither, chroma);
	}
	env->ThrowError("ConvertToBits: unsupported bit depth");
	return nullptr;
}

static BitDepthConvFuncPtr2 get_cuda_conv_bits(int pixelsize, int bits_per_pixel, int tgt_bits, bool dither, bool chroma, PNeoEnv env)
{
	if (pixelsize == 1) {
		return get_cuda_conv_bits<8>(tgt_bits, dither, chroma, env);
	}
	else if (pixelsize == 2) {
		switch (bits_per_pixel) {
		case 10:
			return get_cuda_conv_bits<10>(tgt_bits, dither, chroma, env);
		case 12:
			return get_cuda_conv_bits<12>(tgt_bits, dither, chroma, env);
		case 14:
			return get_cuda_conv_bits<14>(tgt_bits, dither, chroma, env);
		case 16:
			return get_cuda_conv_bits<16>(tgt_bits, dither, chroma, env);
		}
	}
	else if(pixelsize == 4) {
		return get_cuda_conv_bits<32>(tgt_bits, dither, chroma, env);
	}
	env->ThrowError("ConvertToBits: unsupported bit depth");
	return nullptr;
}

ConvertBits::ConvertBits(PClip _child, const int _dither_mode, const int _target_bitdepth, bool _truerange, bool _fulls, bool _fulld, int _dither_bitdepth, IScriptEnvironment* env) :
	GenericVideoFilter(_child), dither_mode(_dither_mode), target_bitdepth(_target_bitdepth), truerange(_truerange),
	fulls(_fulls), fulld(_fulld), dither_bitdepth(_dither_bitdepth)
{

	pixelsize = vi.ComponentSize();
	bits_per_pixel = vi.BitsPerComponent();
	format_change_only = false;

	bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
	bool sse4 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
	bool avx = !!(env->GetCPUFlags() & CPUF_AVX);
	bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);

	BitDepthConvFuncPtr conv_function_full_scale;
	BitDepthConvFuncPtr conv_function_full_scale_no_dither;
	BitDepthConvFuncPtr conv_function_shifted_scale;

	conv_function_chroma = nullptr; // used only for 32bit float

	// CUDA
	conv_cuda = get_cuda_conv_bits(pixelsize, bits_per_pixel, target_bitdepth, dither_mode >= 0, false, env);
	conv_cuda_chroma = get_cuda_conv_bits(pixelsize, bits_per_pixel, target_bitdepth, dither_mode >= 0, true, env);

  if (bits_per_pixel < 32 && target_bitdepth < 32) {
    // 32 bit source: fulls, fulld handled properly
    if (fulls != fulld)
      env->ThrowError("ConvertBits: fulls and fulld should be the same for non-32bit float formats");
  }

  // 8-16bit->32bits support fulls fulld, alpha is always full-full
#define convert_uintN_to_float_functions(uint_X_t, source_bits) \
      conv_function_a = convert_uintN_to_float_c<uint_X_t, source_bits, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, source_bits, false, true, true>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, source_bits, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, source_bits, false, true, false>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, source_bits, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, source_bits, false, false, true>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, source_bits, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = convert_uintN_to_float_c<uint_X_t, source_bits, false, false, false>; \
        conv_function_chroma = convert_uintN_to_float_c<uint_X_t, source_bits, true, false, false>; \
      }

  // ConvertToFloat
  if (target_bitdepth == 32) {
    if (pixelsize == 1) // 8->32 bit
    {
      convert_uintN_to_float_functions(uint8_t, 8)
    }
    else if (pixelsize == 2) // 16->32 bit
    {
      if (vi.IsPlanar() && truerange)
      {
        switch (bits_per_pixel)
        {
        case 10: 
          convert_uintN_to_float_functions(uint16_t, 10);
          break;
        case 12: 
          convert_uintN_to_float_functions(uint16_t, 12);
          break;
        case 14: 
          convert_uintN_to_float_functions(uint16_t, 14);
          break;
        case 16: 
          convert_uintN_to_float_functions(uint16_t, 16);
          break;
        default: env->ThrowError("ConvertToFloat: unsupported bit depth");
        }
      }
      else {
        convert_uintN_to_float_functions(uint16_t, 16);
      }
    }
    else
      env->ThrowError("ConvertToFloat: internal error 32->32 is not valid here");

    conv_function_a = conv_function; 

		if (vi.NumComponents() == 1)
			vi.pixel_type = VideoInfo::CS_Y32;
		else if (vi.Is420())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA420PS : VideoInfo::CS_YUV420PS;
		else if (vi.Is422())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA422PS : VideoInfo::CS_YUV422PS;
		else if (vi.Is444())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA444PS : VideoInfo::CS_YUV444PS;
		else if (vi.IsPlanarRGB())
			vi.pixel_type = VideoInfo::CS_RGBPS;
		else if (vi.IsPlanarRGBA())
			vi.pixel_type = VideoInfo::CS_RGBAPS;
		else
			env->ThrowError("ConvertToFloat: unsupported color space");

		return;
	}
	// ConvertToFloat end

// 32bit->8-16bits support fulls fulld
#define convert_32_to_uintN_functions(uint_X_t, target_bits) \
      conv_function_a = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, false, true, true> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, false, true, true> : convert_32_to_uintN_c<uint_X_t, target_bits, false, true, true>; /* full-full */ \
      if (fulls && fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, false, true, true> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, false, true, true> : convert_32_to_uintN_c<uint_X_t, target_bits, false, true, true>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, true, true, true> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, true, true, true> : convert_32_to_uintN_c<uint_X_t, target_bits, true, true, true>; \
      } \
      else if (fulls && !fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, false, true, false> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, false, true, false> : convert_32_to_uintN_c<uint_X_t, target_bits, false, true, false>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, true, true, false> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, true, true, false> : convert_32_to_uintN_c<uint_X_t, target_bits, true, true, false>; \
      } \
      else if (!fulls && fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, false, false, true> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, false, false, true> : convert_32_to_uintN_c<uint_X_t, target_bits, false, false, true>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, true, false, true> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, true, false, true> : convert_32_to_uintN_c<uint_X_t, target_bits, true, false, true>; \
      } \
      else if (!fulls && !fulld) { \
        conv_function = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, false, false, false> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, false, false, false> : convert_32_to_uintN_c<uint_X_t, target_bits, false, false, false>; \
        conv_function_chroma = avx2 ? convert_32_to_uintN_avx2<uint_X_t, target_bits, true, false, false> : sse4 ? convert_32_to_uintN_sse<uint_X_t, target_bits, true, false, false> : convert_32_to_uintN_c<uint_X_t, target_bits, true, false, false>; \
      }

	// ConvertTo16bit() (10, 12, 14, 16)
	// Conversion to uint16_t targets
	// planar YUV(A) and RGB(A):
	//   from 8 bit -> 10/12/14/16 with strict range expansion or expansion to 16
	//   from 10/12/14 -> 16 bit with strict source range (expansion from 10/12/14 to 16 bit) or just casting pixel_type
	//   from 16 bit -> 10/12/14 bit with strict target range (reducing range from 16 bit to 10/12/14 bits) or just casting pixel_type
	//   from float -> 10/12/14/16 with strict range expansion or expansion to 16
	// packed RGB:
	//   RGB24->RGB48, RGB32->RGB64
	if (target_bitdepth > 8 && target_bitdepth <= 16) {
		// 8,10-16,32 -> 16 bit
		if (pixelsize == 1) // 8->10-12-14-16 bit
		{
			if (truerange)
			{
				switch (target_bitdepth)
				{
				case 10:
					conv_function_full_scale = sse2 ? convert_rgb_8_to_uint16_sse2<10> : convert_rgb_8_to_uint16_c<10>;
					conv_function_shifted_scale = sse2 ? convert_8_to_uint16_sse2<10> : convert_8_to_uint16_c<10>;
					break;
				case 12:
					conv_function_full_scale = sse2 ? convert_rgb_8_to_uint16_sse2<12> : convert_rgb_8_to_uint16_c<12>;
					conv_function_shifted_scale = sse2 ? convert_8_to_uint16_sse2<12> : convert_8_to_uint16_c<12>;
					break;
				case 14:
					conv_function_full_scale = sse2 ? convert_rgb_8_to_uint16_sse2<14> : convert_rgb_8_to_uint16_c<14>;
					conv_function_shifted_scale = sse2 ? convert_8_to_uint16_sse2<14> : convert_8_to_uint16_c<14>;
					break;
				case 16:
					conv_function_full_scale = sse2 ? convert_rgb_8_to_uint16_sse2<16> : convert_rgb_8_to_uint16_c<16>;
					conv_function_shifted_scale = sse2 ? convert_8_to_uint16_sse2<16> : convert_8_to_uint16_c<16>;
					break;
				default: env->ThrowError("ConvertTo16bit: unsupported bit depth");
				}
			}
			else {
				conv_function_full_scale = sse2 ? convert_rgb_8_to_uint16_sse2<16> : convert_rgb_8_to_uint16_c<16>;
				conv_function_shifted_scale = sse2 ? convert_8_to_uint16_sse2<16> : convert_8_to_uint16_c<16>;
			}

			if (fulls)
				conv_function = conv_function_full_scale; // rgb default, RGB scaling is not shift by 2/4/6/8 as in YUV but like 0..255->0..65535
			else
				conv_function = conv_function_shifted_scale; // yuv default

			conv_function_a = conv_function_full_scale; // alpha copy is the same full scale
		}
		else if (pixelsize == 2)
		{
			// 10-16->10-16
			if (truerange)
			{

				// full_scale is used for alpha plane always (keep max opacity 255, 1023, 4095, 16383, 65535)

				// fill conv_function_full_scale and conv_function_shifted_scale
				// first get full_scale converter functions, normal and optional dithered
				if (bits_per_pixel >= target_bitdepth) // reduce range or dither down keeping bit-depth format
				{
					conv_function_full_scale = nullptr; // BitBlt in GetFrame

					if (bits_per_pixel == 16) { // 16->10/12/14 keep full range
						switch (target_bitdepth)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<16, 10, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<16, 10, false> : convert_rgb_uint16_to_uint16_c<16, 10>;
							break;
						case 12: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<16, 12, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<16, 12, false> : convert_rgb_uint16_to_uint16_c<16, 12>;
							break;
						case 14: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<16, 14, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<16, 14, false> : convert_rgb_uint16_to_uint16_c<16, 14>;
							break;
						}
					}
					else if (bits_per_pixel == 14) { // 14->10/12 keep full range
						switch (target_bitdepth)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<14, 10, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<14, 10, false> : convert_rgb_uint16_to_uint16_c<14, 10>;
							break;
						case 12: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<14, 12, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<14, 12, false> : convert_rgb_uint16_to_uint16_c<14, 12>;
							break;
						}
					}
					else if (bits_per_pixel == 12) { // 12->10 keep full range
						switch (target_bitdepth)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<12, 10, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<12, 10, false> : convert_rgb_uint16_to_uint16_c<12, 10>;
							break;
						}
					}

					conv_function_full_scale_no_dither = conv_function_full_scale; // save ditherless, used for possible alpha

					if (dither_mode >= 0) {
						conv_function_full_scale = get_convert_to_16_16_down_dither_function(true /*full scale*/, bits_per_pixel, target_bitdepth, dither_mode, dither_bitdepth, 1/*rgb_step n/a*/, 0 /*cpu none*/);
					}
				}
				else {// expand
							// no dither here
					if (target_bitdepth == 16) { // 10/12/14->16 keep full range
						switch (bits_per_pixel)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<10, 16, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<10, 16, false> : convert_rgb_uint16_to_uint16_c<10, 16>;
							break;
						case 12: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<12, 16, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<12, 16, false> : convert_rgb_uint16_to_uint16_c<12, 16>;
							break;
						case 14: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<14, 16, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<14, 16, false> : convert_rgb_uint16_to_uint16_c<14, 16>;
							break;
						}
					}
					else if (target_bitdepth == 14) { // 10/12->14 keep full range
						switch (bits_per_pixel)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<10, 14, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<10, 14, false> : convert_rgb_uint16_to_uint16_c<10, 14>;
							break;
						case 12: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<12, 14, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<12, 14, false> : convert_rgb_uint16_to_uint16_c<12, 14>;
							break;
						}
					}
					else if (target_bitdepth == 12) { // 10->12 keep full range
						switch (bits_per_pixel)
						{
						case 10: conv_function_full_scale = sse4 ? convert_rgb_uint16_to_uint16_sse2<10, 12, true> : sse2 ? convert_rgb_uint16_to_uint16_sse2<10, 12, false> : convert_rgb_uint16_to_uint16_c<10, 12>;
							break;
						}
					}

					conv_function_full_scale_no_dither = conv_function_full_scale; // save ditherless, used for possible alpha
				}

				// fill shift_range converter functions
				if (bits_per_pixel >= target_bitdepth) { // reduce range 16->14/12/10 14->12/10 12->10. template: bitshift
					if (dither_mode < 0) {
						switch (bits_per_pixel - target_bitdepth)
						{
						case 2:
							conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<false, 2> : avx ? convert_uint16_to_uint16_c_avx<false, 2> : (sse2 ? convert_uint16_to_uint16_sse2<false, 2> : convert_uint16_to_uint16_c<false, 2>);
							break;
						case 4:
							conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<false, 4> : avx ? convert_uint16_to_uint16_c_avx<false, 4> : (sse2 ? convert_uint16_to_uint16_sse2<false, 4> : convert_uint16_to_uint16_c<false, 4>);
							break;
						case 6:
							conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<false, 6> : avx ? convert_uint16_to_uint16_c_avx<false, 6> : (sse2 ? convert_uint16_to_uint16_sse2<false, 6> : convert_uint16_to_uint16_c<false, 6>);
							break;
						}
					}
					else {
						// dither
						conv_function_shifted_scale = get_convert_to_16_16_down_dither_function(false /*not full scale*/, bits_per_pixel, target_bitdepth, dither_mode, dither_bitdepth, 1/*rgb_step n/a*/, 0 /*cpu none*/);
					}
				}
				else { // expand range
					switch (target_bitdepth - bits_per_pixel)
					{
					case 2: conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<true, 2> : avx ? convert_uint16_to_uint16_c_avx<true, 2> : (sse2 ? convert_uint16_to_uint16_sse2<true, 2> : convert_uint16_to_uint16_c<true, 2>); break;
					case 4: conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<true, 4> : avx ? convert_uint16_to_uint16_c_avx<true, 4> : (sse2 ? convert_uint16_to_uint16_sse2<true, 4> : convert_uint16_to_uint16_c<true, 4>); break;
					case 6: conv_function_shifted_scale = avx2 ? convert_uint16_to_uint16_c_avx2<true, 6> : avx ? convert_uint16_to_uint16_c_avx<true, 6> : (sse2 ? convert_uint16_to_uint16_sse2<true, 6> : convert_uint16_to_uint16_c<true, 6>); break;
					}
				}
			}
			else {
				// no conversion for truerange == false
			}

			// 10/12/14 -> 16 bit or 16 bit -> 10/12/14 bit
			// range reducing or expansion (truerange=true), or just overriding the pixel_type, keeping scale at 16 bits
			// 10-16 -> 10->16 truerange == false already handled
			if (truerange) {
				if (fulls)
					conv_function = conv_function_full_scale; // rgb default, RGB scaling is not shift by 2/4/6/8 as in YUV but like 0..255->0..65535
				else
					conv_function = conv_function_shifted_scale; // yuv default

				conv_function_a = conv_function_full_scale_no_dither; // alpha copy is always full scale w/o dithering
			}
			else { // truerange==false
						 // 10->12 .. 16->12 etc
						 // only vi bit_depth format override
				format_change_only = true;
			}
		}
    else if (pixelsize == 4) // 32->10-16 bit
		{
			if (truerange) {
				switch (target_bitdepth)
				{
				case 10:
          convert_32_to_uintN_functions(uint16_t, 10); // all variations of fulls fulld
					break;
				case 12:
          convert_32_to_uintN_functions(uint16_t, 12);
					break;
				case 14:
          convert_32_to_uintN_functions(uint16_t, 14);
					break;
				case 16:
          convert_32_to_uintN_functions(uint16_t, 16);
					break;
				}
			}
			else {
        convert_32_to_uintN_functions(uint16_t, 16);
			}
		}
		else {
			env->ThrowError("ConvertTo16bit: unsupported bit depth");
		}

		// set output vi format
		if (vi.IsRGB24()) {
			if (target_bitdepth == 16)
				vi.pixel_type = VideoInfo::CS_BGR48;
			else
				env->ThrowError("ConvertTo16bit: unsupported bit depth");
		}
		else if (vi.IsRGB32()) {
			if (target_bitdepth == 16)
				vi.pixel_type = VideoInfo::CS_BGR64;
			else
				env->ThrowError("ConvertTo16bit: unsupported bit depth");
		}
		else {
			// Y or YUV(A) or PlanarRGB(A)
			if (vi.IsYV12()) // YV12 can have an exotic compatibility constant
				vi.pixel_type = VideoInfo::CS_YV12;
			int new_bitdepth_bits;
			switch (target_bitdepth) {
			case 8: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_8; break;
			case 10: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_10; break;
			case 12: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_12; break;
			case 14: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_14; break;
			case 16: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_16; break;
			case 32: new_bitdepth_bits = VideoInfo::CS_Sample_Bits_32; break;
			}
			vi.pixel_type = (vi.pixel_type & ~VideoInfo::CS_Sample_Bits_Mask) | new_bitdepth_bits;
		}

		return;
	}

	// ConvertTo8bit()
	if (target_bitdepth == 8) {
		if (pixelsize == 2) // 16(,14,12,10)->8 bit
		{
			// it gets complicated, so we better using tuples for function lookup
			// parameters for full scale: source bitdepth, dither_type (-1:none, 0:ordered), target_dither_bitdepth(default 8, 2,4,6), rgb_step(3 for RGB48, 4 for RGB64, 1 for all planars)
			// rgb_step can differ from 1 only when source bits_per_pixel==16 and packed RGB type
			// target_dither_bitdepth==8 (RFU for dithering down from e.g. 10->2 bit)

			if (dither_mode == 0 && (dither_bitdepth != 2 && dither_bitdepth != 4 && dither_bitdepth != 6 && dither_bitdepth != 8))
				env->ThrowError("ConvertBits: invalid dither target bitdepth %d", dither_bitdepth);

			// fill conv_function_full_scale and conv_function_shifted_scale
			// conv_function_full_scale_no_dither: for alpha plane
			if (truerange) {
				conv_function_full_scale = get_convert_to_8_function(true, bits_per_pixel, dither_mode, dither_bitdepth, 1, CPUF_SSE2);
				conv_function_full_scale_no_dither = get_convert_to_8_function(true, bits_per_pixel, -1, dither_bitdepth /* n/a */, 1, CPUF_SSE2); // force dither_mode==-1
				conv_function_shifted_scale = get_convert_to_8_function(false, bits_per_pixel, dither_mode, dither_bitdepth, 1, CPUF_SSE2);
			}
			else {
				conv_function_full_scale = get_convert_to_8_function(true, 16, dither_mode, dither_bitdepth, 1, CPUF_SSE2);
				conv_function_full_scale_no_dither = get_convert_to_8_function(true, 16, -1, dither_bitdepth /* n/a */, 1, CPUF_SSE2);
				conv_function_shifted_scale = get_convert_to_8_function(false, 16, dither_mode, dither_bitdepth, 1, CPUF_SSE2);
			}

			// override for RGB48 and 64 (internal rgb_step may differ when dithering is used
			if (vi.IsRGB48()) { // packed RGB: specify rgb_step 3 or 4 for dither table access
				conv_function_full_scale = get_convert_to_8_function(true, 16, dither_mode, dither_bitdepth, 3, CPUF_SSE2);
      } else if(vi.IsRGB64()) {
				conv_function_full_scale = get_convert_to_8_function(true, 16, dither_mode, dither_bitdepth, 4, CPUF_SSE2);
			}

			// packed RGB scaling is full_scale 0..65535->0..255
			if (fulls)
				conv_function = conv_function_full_scale; // rgb default, RGB scaling is not shift by 2/4/6/8 as in YUV but like 0..255->0..65535
			else
				conv_function = conv_function_shifted_scale; // yuv default

			conv_function_a = conv_function_full_scale_no_dither; // alpha copy is the same full scale, w/o dithering

		}
		else if (vi.ComponentSize() == 4) // 32->8 bit, no dithering option atm
		{
      convert_32_to_uintN_functions(uint8_t, 8); // all combinations of fulls, fulld
		}
		else
			env->ThrowError("ConvertTo8bit: unsupported bit depth");

		if (vi.NumComponents() == 1)
			vi.pixel_type = VideoInfo::CS_Y8;
		else if (vi.Is420())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA420 : VideoInfo::CS_YV12;
		else if (vi.Is422())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA422 : VideoInfo::CS_YV16;
		else if (vi.Is444())
			vi.pixel_type = vi.IsYUVA() ? VideoInfo::CS_YUVA444 : VideoInfo::CS_YV24;
		else if (vi.IsRGB48())
			vi.pixel_type = VideoInfo::CS_BGR24;
		else if (vi.IsRGB64())
			vi.pixel_type = VideoInfo::CS_BGR32;
		else if (vi.IsPlanarRGB())
			vi.pixel_type = VideoInfo::CS_RGBP;
		else if (vi.IsPlanarRGBA())
			vi.pixel_type = VideoInfo::CS_RGBAP;
		else
			env->ThrowError("ConvertTo8bit: unsupported color space");

		return;
	}

	env->ThrowError("ConvertBits: unsupported target bit-depth (%d)", target_bitdepth);

}

AVSValue __cdecl ConvertBits::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
	PClip clip = args[0].AsClip();
	//0   1        2        3         4         5           6    
	//c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b

	const VideoInfo &vi = clip->GetVideoInfo();

	int create_param = (int)reinterpret_cast<intptr_t>(user_data);

	// bits parameter is compulsory
	if (!args[1].Defined() && create_param == 0) {
		env->ThrowError("ConvertBits: missing bits parameter");
	}

	// when converting from/true 10-16 bit formats, truerange=false indicates bitdepth of 16 bits regardless of the 10-12-14 bit format
	bool assume_truerange = args[2].AsBool(true); // n/a for non planar formats
																								// bits parameter

	int target_bitdepth = args[1].AsInt(create_param); // default comes by calling from old To8,To16,ToFloat functions
	int source_bitdepth = vi.BitsPerComponent();
	int dither_bitdepth = args[4].AsInt(target_bitdepth);

	if (target_bitdepth != 8 && target_bitdepth != 10 && target_bitdepth != 12 && target_bitdepth != 14 && target_bitdepth != 16 && target_bitdepth != 32)
		env->ThrowError("ConvertBits: invalid bit depth: %d", target_bitdepth);

	if (create_param == 8 && target_bitdepth != 8)
		env->ThrowError("ConvertTo8Bit: invalid bit depth: %d", target_bitdepth);
	if (create_param == 32 && target_bitdepth != 32)
		env->ThrowError("ConvertToFloat: invalid bit depth: %d", target_bitdepth);
	if (create_param == 16 && (target_bitdepth == 8 || target_bitdepth == 32))
		env->ThrowError("ConvertTo16bit: invalid bit depth: %d", target_bitdepth);

	if (args[2].Defined()) {
		if (!vi.IsPlanar())
			env->ThrowError("ConvertBits: truerange specified for non-planar source");
	}

	// override defaults, e.g. set full range for greyscale clip conversion that is RGB
  // Post 2664: can be set. Full range is default also for float (and cannot be set to false)
  bool fulls = args[5].AsBool(vi.IsRGB()/* || ((target_bitdepth == 32 || source_bitdepth == 32))*/);
	bool fulld = args[6].AsBool(fulls);

	int dither_type = args[3].AsInt(-1);
	bool dither_defined = args[3].Defined();
	if (dither_defined && dither_type != 1 && dither_type != 0 && dither_type != -1)
    env->ThrowError("ConvertBits: invalid dither type parameter. Only -1 (disabled), 0 (ordered dither) or 1 (Floyd-S) is allowed");

	if (dither_type >= 0) {
		if (source_bitdepth < target_bitdepth)
			env->ThrowError("ConvertBits: dithering is allowed only for scale down");
		if (dither_bitdepth > target_bitdepth)
			env->ThrowError("ConvertBits: dither_bits must be <= target bitdepth");
		if (target_bitdepth == 32)
			env->ThrowError("ConvertBits: dithering is not allowed only for 32 bit targets");
	}

	if (dither_type == 0) {

		if (source_bitdepth == 32)
			env->ThrowError("ConvertBits: dithering is not allowed only for 32 bit sources");

		if (dither_bitdepth < 2 || dither_bitdepth > 16)
			env->ThrowError("ConvertBits: invalid dither_bits specified");

		if (dither_bitdepth % 2)
			env->ThrowError("ConvertBits: dither_bits must be even");

		if (source_bitdepth - dither_bitdepth > 8)
			env->ThrowError("ConvertBits: dither_bits cannot differ with more than 8 bits from source");

		if (source_bitdepth == 8)
			env->ThrowError("ConvertBits: dithering is not supported for 8 bit sources");
	}

	// floyd
	if (dither_type == 1) {

		if (source_bitdepth == 8 || source_bitdepth == 32)
			env->ThrowError("ConvertBits: Floyd-S: dithering is allowed only for 10-16 bit sources");

		if (dither_bitdepth < 0 || dither_bitdepth > 16)
			env->ThrowError("ConvertBits: Floyd-S: invalid dither_bits specified");

		if ((dither_bitdepth > 8 && (dither_bitdepth % 2) != 0)) // must be even above 8 bits. 0 is ok, means real b/w
			env->ThrowError("ConvertBits: Floyd-S: dither_bits must be 0..8, 10, 12, 14, 16");
	}

	// no change -> return unmodified if no dithering required, or dither bitdepth is the same as target
	if (source_bitdepth == target_bitdepth) { // 10->10 .. 16->16
		if (dither_type < 0 || dither_bitdepth == target_bitdepth)
			return clip;
		if (vi.IsRGB() && !vi.IsPlanar())
			env->ThrowError("ConvertBits: dithering_bits should be the as target bitdepth for packed RGB formats");
		// here: we allow e.g. a 16->16 bit conversion with dithering bitdepth of 8
	}

	// YUY2 conversion is limited
	if (vi.IsYUY2()) {
		env->ThrowError("ConvertBits: YUY2 source is 8-bit only");
	}

	if (vi.IsYV411()) {
		env->ThrowError("ConvertBits: YV411 source cannot be converted");
	}

	// packed RGB conversion is limited
	if (vi.IsRGB24() || vi.IsRGB32()) {
		if (target_bitdepth != 16)
			env->ThrowError("ConvertBits: invalid bit-depth specified for packed RGB");
	}

	if (vi.IsRGB48() || vi.IsRGB64()) {
		if (target_bitdepth != 8)
			env->ThrowError("ConvertBits: invalid bit-depth specified for packed RGB");
	}

	// remark
	// source_10_bit.ConvertTo16bit(truerange=true)  : upscale range
	// source_10_bit.ConvertTo16bit(truerange=false) : leaves data, only format conversion
	// source_10_bit.ConvertTo16bit(bits=12,truerange=true)  : upscale range from 10 to 12
	// source_10_bit.ConvertTo16bit(bits=12,truerange=false) : leaves data, only format conversion
	// source_16_bit.ConvertTo16bit(bits=10, truerange=true)  : downscale range
	// source_16_bit.ConvertTo16bit(bits=10, truerange=false) : leaves data, only format conversion

  if (fulls != fulld && target_bitdepth != 32 && source_bitdepth != 32)
    env->ThrowError("ConvertBits: fulls must be the same as fulld for non 32bit target and source");

	return new ConvertBits(clip, dither_type, target_bitdepth, assume_truerange, fulls, fulld, dither_bitdepth, env);
}


PVideoFrame __stdcall ConvertBits::GetFrame(int n, IScriptEnvironment* env_) 
{
	PNeoEnv env = env_;
	PVideoFrame src = child->GetFrame(n, env);

	if (format_change_only)
	{
		// for 10-16 bit: simple format override in constructor
		return src;
	}

	PVideoFrame dst = env->NewVideoFrame(vi);

	if (vi.IsPlanar())
	{
		int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
		int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
		int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
		for (int p = 0; p < vi.NumComponents(); ++p) {
			const int plane = planes[p];
			if (IS_CUDA) {
				const bool chroma = (plane == PLANAR_U || plane == PLANAR_V);
				(chroma ? conv_cuda_chroma : conv_cuda)(
					src->GetReadPtr(plane), dst->GetWritePtr(plane),
					src->GetRowSize(plane), src->GetHeight(plane),
					src->GetPitch(plane), dst->GetPitch(plane), env);
				DEBUG_SYNC;
			}
			else if (plane == PLANAR_A) {
				if (conv_function_a == nullptr)
					env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
				else
					conv_function_a(src->GetReadPtr(plane), dst->GetWritePtr(plane),
						src->GetRowSize(plane), src->GetHeight(plane),
						src->GetPitch(plane), dst->GetPitch(plane));
			}
			else if (conv_function == nullptr)
				env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane), src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
			else {
				const bool chroma = (plane == PLANAR_U || plane == PLANAR_V);
				if (chroma && conv_function_chroma != nullptr)
					// 32bit float needs separate conversion (possible chroma -0.5 .. 0.5 option)
					// until then the conv_function_ch behaves the same as conv_function
          // see #ifdef FLOAT_CHROMA_IS_HALF_CENTERED
					conv_function_chroma(src->GetReadPtr(plane), dst->GetWritePtr(plane),
						src->GetRowSize(plane), src->GetHeight(plane),
						src->GetPitch(plane), dst->GetPitch(plane));
				else
					conv_function(src->GetReadPtr(plane), dst->GetWritePtr(plane),
						src->GetRowSize(plane), src->GetHeight(plane),
						src->GetPitch(plane), dst->GetPitch(plane));
			}
		}
	}
	else {
		// packed RGBs
		if (IS_CUDA) {
			conv_cuda(src->GetReadPtr(), dst->GetWritePtr(),
				src->GetRowSize(), src->GetHeight(),
				src->GetPitch(), dst->GetPitch(), env);
			DEBUG_SYNC;
		}
		else {
			conv_function(src->GetReadPtr(), dst->GetWritePtr(),
				src->GetRowSize(), src->GetHeight(),
				src->GetPitch(), dst->GetPitch());
		}
	}
	return dst;
}

