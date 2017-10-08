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

src = LWLibavVideoSource("test.ts").Trim(0,4999)
#return src

p60 = src.OnCPU(0).KTGMC(Preset="Fast").OnCUDA(0)
stt = src.KAnalyzeStatic(30, 15)
#stt.KShowStatic(src)
p60 = p60.KMergeStatic(src,stt)

fm = src.KFMFrameAnalyze(15, 7, 20, 8).KFMCycleAnalyze(src)
combe = src.KTelecine(fm).KRemoveCombe(30, 50, 150, 0, 5)
p24 = combe

p60.KFMSwitch(p24, fm, combe, 2, show=True)

#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)
