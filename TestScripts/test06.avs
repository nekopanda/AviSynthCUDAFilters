srcfile = "test.ts"
#srcfile = "T:\sandbox\t28\05\04_hybridheart.ts"
LWLibavVideoSource(srcfile)
AssumeTFF()

#return QTGMC(Preset="Faster").Prefetch(4)

fm = KFM()
#BlankClip(last).KTCMerge(KIVTC(fm),fm)
#nnedi3(-2).KTCMerge(KIVTC(fm),fm)
QTGMC(Preset="Faster").KTCMerge(KIVTC(fm),fm)

#TFM(mode=0, order=-1, pp=1, slow=2, mChroma=true, display=true)
#TDecimate(mode=1)

Prefetch(4)