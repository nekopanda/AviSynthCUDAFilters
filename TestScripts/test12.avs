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

function cf2460(clip c) {
    return c.SelectEvery(4, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3)
}
function cf3060(clip c) {
    return c.SelectEvery(1, 0, 0)
}

src = LWLibavVideoSource("test.ts")
srcg = src.OnCPU(0)
#return src

#bobbed = src.QTGMC(Preset="Medium")
#bobbed = src.OnCPU(0).KTGMC(Preset="Medium").OnCUDA(2)

#return src.TDeint( mode=1, order=-1, field=-1, edeint=src.nnedi3(field=-2), emask=src.TMM2(mode=1, order=-1, field=-1) ).Prefetch(4)

p60 =srcg.KTGMC(Preset="Faster")
#p60 = src.QTGMC(EdiMode="TDeint")
#p60 = src.Bob()
#p30 = src.KRemoveCombe(30, 50, 70, 0, 5)
stt = srcg.KAnalyzeStatic(30, 15)
#return src
#stt.KShowStatic(src)
p60 = p60.KMergeStatic(srcg,stt)

fm = srcg.KFMFrameAnalyze(15, 7, 20, 8).OnCUDA(0).KFMCycleAnalyze(src).OnCPU(0)
tc = srcg.KTelecine(fm)
combe = tc.KRemoveCombe(30, 100, 100, 0, 0, show=false)
#src.OnCPU(2).KTGMC(Preset="Fast").OnCUDA(2).KFMSwitch(p24, fm, p24, 2, show=True)

p24 = combe.KSMDegrain()
#p24 = combe

#p60 = p60.AssertOnCUDA().OnCUDA(0)
#p24 = p24.AssertOnCUDA().OnCUDA(0)
#fm = fm.AssertOnCUDA().OnCUDA(0)
#combe = combe.AssertOnCUDA().OnCUDA(0)

p60.KFMSwitch(p24, fm, combe, 0.5, show=True).OnCUDA(0)

#DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)


#Interleave(c1.KShowCombe().Subtitle("C1"), c3.Subtitle("C3"))
#Interleave(p24.Subtitle("24ORIG"), c3x2.KShowCombe().Subtitle("C3x2"), c3.Subtitle("C3"))
#c3x2.KShowCombe()
#return src

#KMergeDev(bobbed, src, 8, 0)
#AudioDub(LWLibavAudioSource("news.ts"))
#Prefetch(6)

#Interleave(last, bobbed.Subtitle("KTGMC"))

#AudioDub(LWLibavAudioSource("test.ts"))
