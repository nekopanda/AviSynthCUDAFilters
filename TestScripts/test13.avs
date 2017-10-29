#SetGraphAnalysis(true)
SetLogParams("avsp.log", LOG_WARNING)
SetMemoryMax(8192, type=DEV_TYPE_CUDA)
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

src = LWLibavVideoSource("test.ts")
srcg = src.OnCPU(2)

#p60 = srcg.KTGMC_Bob(0, 0.5)
p60 = srcg.KTGMC()
stt = srcg.KAnalyzeStatic(30, 15)
#return stt.OnCUDA(0).KShowStatic(src)
p60 = p60.KMergeStatic(srcg,stt).OnCUDA(2)
return p60

fm = srcg.KFMFrameAnalyze(15, 7, 20, 8).OnCUDA(2).KFMCycleAnalyze(src)
combe = srcg.KTelecine(fm.OnCPU(2)).KRemoveCombe(30, 50, 150, 0, 5)
p24 = combe.KSMDegrain().OnCUDA(2)

p60.KFMSwitch(p24, fm, combe.OnCUDA(2), 2, show=True)

#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)
