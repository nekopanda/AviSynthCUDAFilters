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

function KSMDegrain_MAnalyze( clip srchSuper, clip srchSuperP, bool isb, int delta, int blksize, int overlap, \
					int search, bool truemotion, bool chroma, int batch)
{
	# partial mv on CPU
	pmv = srchSuperP.KMAnalyse(blksize=blksize, overlap=overlap, search=search, isb=isb, chroma=chroma, \
					delta=delta, meander=false, batch=1).OnCPU(2)

	# main mv on CUDA
	return srchSuper.KMAnalyse(blksize=blksize, overlap=overlap, search=search, isb=isb, chroma=chroma, \
					delta=delta, meander=false, partial=pmv, batch=batch)
}

function KSMDegrain(clip c)
{
    super = c.KTGMC_ToFullRange(U=3,V=3).KMSuper(pel=1,chroma=True)
    psuper = super.KMPartialSuper().OnCUDA(2)
    dsuper = c.KMSuper(pel=1,chroma=True,levels=1)
    bvec2 = super.KSMDegrain_MAnalyze(psuper, True,2,16,8,4,False,True,4)
    bvec1 = super.KSMDegrain_MAnalyze(psuper, True,1,16,8,4,False,True,4)
    fvec1 = super.KSMDegrain_MAnalyze(psuper, False,1,16,8,4,False,True,4)
    fvec2 = super.KSMDegrain_MAnalyze(psuper, False,2,16,8,4,False,True,4)
    return c.KMDegrain2(dsuper, bvec1, fvec1, bvec2, fvec2,thSAD=300,thSADC=150,thSCD1=1600,thSCD2=130,limit=255,limitc=255)
}

src = LWLibavVideoSource("test.ts").Trim(0,4999)
#return src

#bobbed = src.QTGMC(Preset="Medium")
#bobbed = src.OnCPU(0).KTGMC(Preset="Medium").OnCUDA(2)

#return src.TDeint( mode=1, order=-1, field=-1, edeint=src.nnedi3(field=-2), emask=src.TMM2(mode=1, order=-1, field=-1) ).Prefetch(4)

p60 = src.OnCPU(2).KTGMC(Preset="Slower").OnCUDA(2)
#p60 = src.Bob()
#p30 = src.KRemoveCombe(30, 50, 70, 0, 5)
stt = src.KStaticAnalyze2(30, 15)
#return src
#stt.KShowStatic2(src)
p60 = p60.KStaticMerge2(src,stt)

fm = src.KFMFrameAnalyze(15, 7, 20, 8).Prefetch(4).KFMCycleAnalyze(src)
combe = src.KTelecine(fm).KRemoveCombe(30, 50, 150, 0, 5).Prefetch(2)
#src.OnCPU(2).KTGMC(Preset="Fast").OnCUDA(2).KFMSwitch(p24, fm, p24, 2, show=True)

p24 = combe.OnCPU(0).KSMDegrain().OnCUDA(2)

p60.KFMSwitch(p24, fm, combe, 2, show=True)

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
