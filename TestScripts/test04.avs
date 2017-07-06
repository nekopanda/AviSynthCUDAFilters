srcfile = "test.ts"
LWLibavVideoSource(srcfile)

super1 = MSuper(pel=1).Time("Super1")
super2 = MSuper(pel=2).Time("Super2")
vec1 = super1.MAnalyse(isb=false, delta=1, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0).Time("MAnalyze1")
vec2 = super2.MAnalyse(isb=false, delta=1, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0).Time("MAnalyze2")

show1 = super1.MShow(vec1, tol=10000,showsad=true)
show2 = super2.MShow(vec2, tol=10000,showsad=true)

Interleave(show1.Subtitle("ORIGINAL", y=500,size=48), show2.Subtitle("MOD",y=500,size=48))
