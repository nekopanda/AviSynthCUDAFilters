#SetGraphAnalysis(true)
SetLogParams("avsp.log", LOG_WARNING)
SetMemoryMax(2048, type=DEV_TYPE_CUDA)
platform = "x64"
#config = "Debug"
config = "Release"
prefix = "../" + platform + "/" + config + "/"
LoadPlugin(prefix + "KNNEDI3.dll")
LoadPlugin(prefix + "KTGMC.dll")

Import(".\KTGMC.avsi")

src = LWLibavVideoSource("test.ts").Trim(0, 1000)

src.QTGMC(SourceMatch=3, Lossless=2, tr0=1, tr1=1, tr2=1)
#src.OnCPU(2).KTGMC(SourceMatch=3, Lossless=2, tr0=1, tr1=1, tr2=1).OnCUDA(2)
#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)
Prefetch(6)

#AudioDub(LWLibavAudioSource("test.ts"))
