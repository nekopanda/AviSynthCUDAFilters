srcpath = "test.ts"
#srcpath = "T:\sandbox\t28\05\06_tsunagu.ts"
LWLibavVideoSource(srcpath, repeat=true, dominance=1) # .Trim(356,356+300-1)

TFM(mode=6, order=-1, pp=1, slow=2, mChroma=true, display=false)
src = TDecimate(mode=1)

super = src.MSuper()
vecb2 = super.MAnalyse(isb=true, delta=2, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0)
vecb1 = super.MAnalyse(isb=true, delta=1, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0)
vecf1 = super.MAnalyse(isb=false, delta=1, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0)
vecf2 = super.MAnalyse(isb=false, delta=2, blksize=16, overlap=8, pelsearch=2,truemotion=false,lambda=400,pnew=25,global=true,meander=true,sadx264=0)

deg11 = MDegrain1(src, super, vecb1, vecf1)
deg12 = MDegrain1(src, super, vecb2, vecf2)

deg2 = MDegrain2(src, super, vecb1, vecf1, vecb2, vecf2)

deg1 = deg11.Merge(deg12)
deg2 = deg2.Merge(src, 0.167)


#AudioDub(LWLibavAudioSource(srcpath,av_sync=true))

#MShow(vec, tol=10000,showsad=true)

#diff = mt_makediff(my, ref)
diff = mt_makediff(deg1, deg2, u=3, v=3)
# (diff < 128) ? 20 : (diff > 128) ? 200 : 128
diff.mt_lut("x 127 < 20 diff 129 > 200 128 ? ?", u=3, v=3)

knl = src.KNLMeansCL()
smd = src.SMDegrain()

#StackHorizontal(smd.Crop(588, 78, -848, -738).Subtitle("SMDegrain"), src.Crop(588, 78, -848, -738).Subtitle("Source"), knl.Crop(588, 78, -848, -738).Subtitle("KNLMeans"))

return smd
