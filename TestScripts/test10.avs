src = LWLibavVideoSource("test.ts")
#src.NNEDI3(field=-2,nsize=1,nns=1,qual=1,pscrn=2,U=True,V=True)
src.OnCPU(1).KNNEDI3(field=-2,nsize=1,nns=1,qual=1,pscrn=2,U=True,V=True).OnCUDA(1)

