srcfile = "test.ts"
LWLibavVideoSource(srcfile)
AssumeTFF()
TFM(mode=6, order=-1, pp=1, slow=2, mChroma=true, display=false)
TDecimate(mode=1)

ms = MSuper(pel=2,isse=false)
a1 = ms.MAnalyse(isb=true,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=True,meander=True,sadx264=0)
a2 = ms.MAnalyse(isb=true,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=True,meander=False,sadx264=0)
a3 = ms.MAnalyse(isb=true,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=True,meander=False,sadx264=1)

as1 = ms.MShow(a1, showsad=True)
as2 = ms.MShow(a2, showsad=True)
as3 = ms.MShow(a3, showsad=True)

Interleave(as1.Subtitle("ORIGINAL", y=30), as2.Subtitle("COMPAT", y=30), as3.Subtitle("MOD",y=30))
