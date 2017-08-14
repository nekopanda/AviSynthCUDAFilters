srcpath = "test.ts"
LWLibavVideoSource(srcpath)

#MSuper().MAnalyze(isb=true, delta=2, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true)
s = KMSuper()
kap = s.KMPartialSuper().KMAnalyse(isb=true,delta=1,blksize=16,overlap=8,lambda=400,global=true,meander=false)
karef = s.KMAnalyse(isb=true,delta=1,blksize=16,overlap=8,lambda=400,global=true,meander=false,partial=kap)
kacuda = s.OnCPU(0).KMAnalyse(isb=true,delta=1,blksize=16,overlap=8,lambda=400,global=true,meander=false,partial=kap.OnCPU(0)).OnCUDA(0)
KMAnalyzeCheck2(karef, kacuda, last)
