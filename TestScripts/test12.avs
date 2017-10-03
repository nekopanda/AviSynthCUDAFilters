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

src = LWLibavVideoSource("news.ts")
#return src

#bobbed = src.QTGMC(Preset="Medium")
#bobbed = src.OnCPU(0).KTGMC(Preset="Medium").OnCUDA(2)
#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)

KFMDev(src, 5, 100, 100, 1)

#KMergeDev(bobbed, src, 8, 0)
#AudioDub(LWLibavAudioSource("news.ts"))
#Prefetch(6)

#Interleave(last, bobbed.Subtitle("KTGMC"))

#AudioDub(LWLibavAudioSource("test.ts"))
