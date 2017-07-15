srcfile = "test.ts"
LWLibavVideoSource(srcfile)
AssumeTFF()
TFM(mode=6, order=-1, pp=1, slow=2, mChroma=true, display=false)
TDecimate(mode=1)
#return last

ks = KMSuper().Time("Super1")
# check fail if isse=true
ms = MSuper(pel=2,isse=false).Time("Super2")
kb1 = ks.KMAnalyse(isb=true,  delta=1, blksize=16, overlap=8, lambda=400, lsad=400, global=true)
mb1 = ms.MAnalyse(isb=true,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0)
kf1 = ks.KMAnalyse(isb=false,  delta=1, blksize=16, overlap=8, lambda=400, lsad=400, global=true)
mf1 = ms.MAnalyse(isb=false,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0)

#KMSuperCheck(ks, ms, last).Time("KMSuperCheck")
#KMAnalyzeCheck(kb1, mb1, last).Time("KMAnalyzeCheckB")
#KMAnalyzeCheck(kf1, mf1, last).Time("KMAnalyzeCheckF")

kdeg = KMDegrain1(ks, kb1, kf1)
mdeg = MDegrain1(ms, mb1, mf1)

kcomp = KMCompensate(ks, kb1)
mcomp = MCompensate(ms, mb1)

#diff = mt_makediff(my, ref)
diff = mt_makediff(kdeg, mdeg, u=3, v=3)
#diff = mt_makediff(kcomp, mcomp, u=3, v=3)
# (diff < 128) ? 20 : (diff > 128) ? 200 : 128
diff.mt_lut("x 128 < 20 diff 128 > 200 128 ? ?", u=3, v=3)

#return kcomp
#return mcomp

#Interleave(show1.Subtitle("ORIGINAL", y=500,size=48), show2.Subtitle("MOD",y=500,size=48))
