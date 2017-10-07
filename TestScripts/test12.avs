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

src = LWLibavVideoSource("test.ts")
#return src

#bobbed = src.QTGMC(Preset="Medium")
#bobbed = src.OnCPU(0).KTGMC(Preset="Medium").OnCUDA(2)
#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)

#return KFMFrameDev(src, 15, 20, 20, 20)
#return src.Bob()
fm = src.KFMFrameAnalyze(15, 7, 20, 8).Prefetch(4).KFMCycleAnalyze(src)
#src.KTelecine(fm).KRemoveCombe(0.5, 0).KShowCombe()
p24 = src.KTelecine(fm)

c1 = p24.KFindCombe1(0.1)

clip60 = src.OnCPU(0).KTGMC_Bob(0,0.5).KGaussResize(p=2,chroma=True).OnCUDA(0)
c2 = p24.KFindCombe2(clip60, fm, 5)

c3 = p24.KFindCombe3(30, 50, 150, show=true)
c3x2 = p24.KFindCombe3(30, 50, 150, 0, 5, false)

#Interleave(c1.KShowCombe().Subtitle("C1"), c3.Subtitle("C3"))
Interleave(p24.Subtitle("24ORIG"), c3x2.KShowCombe().Subtitle("C3x2"), c3.Subtitle("C3"))
#c3x2.KShowCombe()
#return src

#KMergeDev(bobbed, src, 8, 0)
#AudioDub(LWLibavAudioSource("news.ts"))
Prefetch(6)

#Interleave(last, bobbed.Subtitle("KTGMC"))

#AudioDub(LWLibavAudioSource("test.ts"))
