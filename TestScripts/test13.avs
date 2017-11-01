#SetGraphAnalysis(true)
SetLogParams("avsp.log", LOG_WARNING)
SetMemoryMax(2048, type=DEV_TYPE_CUDA)
platform = "x64"
#config = "Debug"
config = "Release"
configKFM = "Release"
prefix = "../" + platform + "/" + config + "/"
prefixKFM = "../" + platform + "/" + configKFM + "/"
LoadPlugin(prefix + "KNNEDI3.dll")
LoadPlugin(prefix + "KTGMC.dll")
LoadPlugin(prefixKFM + "KFM.dll")

Import(".\KTGMC.avsi")
Import(".\KSMDegrain.avsi") 

src = LWLibavVideoSource("banding.ts")
#return src
srch = src.ConvertBits(12)
srcg = srch.OnCPU(2)

#srcg.KTemporalNR(3, 1).KDeband(25, 1, 1, false).OnCUDA(0)
return srcg.KEdgeLevel(10,25,0,0,true).OnCUDA(0).ConvertBits(8)
#srch
#return src
#src.KTemporalNR(3, 4)
#src.KFMFrameAnalyzeShow()

ConvertBits(8, dither=0)

#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)
