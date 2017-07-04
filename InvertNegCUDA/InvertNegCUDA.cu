#include <windows.h>
#include "avisynth.h"

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

int nblocks(int n, int block) {
	return (n + block - 1) / block;
}

class InvertNeg : public GenericVideoFilter {
public:
	InvertNeg(PClip _child, IScriptEnvironment* env);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};

InvertNeg::InvertNeg(PClip _child, IScriptEnvironment* env) :
GenericVideoFilter(_child) {
	if (!vi.IsPlanar() || !vi.IsYUV()) {
		env->ThrowError("InvertNeg: planar YUV data only!");
	}
}

__global__ void InvertNegKernel(
	const unsigned char* srcp, unsigned char* dstp,
	int src_pitch, int dst_pitch, int row_size, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < row_size && y < height) {
		dstp[x + y * dst_pitch] = srcp[x + y * src_pitch] ^ 255;
	}
}

PVideoFrame __stdcall InvertNeg::GetFrame(int n, IScriptEnvironment* env) {

	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);

	const unsigned char* srcp;
	unsigned char* dstp;
	int src_pitch, dst_pitch, row_size, height;
	int p, x, y;

	int planes[] = { PLANAR_Y, PLANAR_V, PLANAR_U };

	for (p = 0; p<3; p++) {
		srcp = src->GetReadPtr(planes[p]);
		dstp = dst->GetWritePtr(planes[p]);

		src_pitch = src->GetPitch(planes[p]);
		dst_pitch = dst->GetPitch(planes[p]);
		row_size = dst->GetRowSize(planes[p]);
		height = dst->GetHeight(planes[p]);

		if (src->IsCUDA()) {
			//
			dim3 threads(32, 16);
			dim3 blocks(nblocks(row_size, threads.x), nblocks(height, threads.y));
			InvertNegKernel << <blocks, threads >> >(srcp, dstp, src_pitch, dst_pitch, row_size, height);
		}
		else{
			for (y = 0; y < height; y++) {
				for (x = 0; x < row_size; x++) {
					dstp[x] = srcp[x] ^ 255;
				}
				srcp += src_pitch; // or srcp = srcp + src_pitch;
				dstp += dst_pitch; // or dstp = dstp + dst_pitch;
			}
		}

	}
	return dst;
}

AVSValue __cdecl Create_InvertNeg(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new InvertNeg(args[0].AsClip(), env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("InvertNeg", "c", Create_InvertNeg, 0);
	return "InvertNeg sample plugin";
}
