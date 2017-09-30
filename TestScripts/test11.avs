SetGraphAnalysis(true)
SetMemoryMax(2048, type=DEV_TYPE_CUDA)
platform = "x64"
#config = "Debug"
config = "Release"
prefix = "../" + platform + "/" + config + "/"
LoadPlugin(prefix + "KNNEDI3.dll")
LoadPlugin(prefix + "KTGMC.dll")

Import(".\KTGMC.avsi")
#Import(".\QTGMC_Ref.avsi")

preset = "Fast"

src = LWLibavVideoSource("test.ts")

# prevent NNEDI3 return wrong parity (GetParity is not implemented correctly in NNEDI3.)
src = src.AssumeTFF()

ref = src.QTGMC(Preset=preset, SourceMatch=3, Lossless=2)
my = src.OnCPU(0).KTGMC(Preset=preset, SourceMatch=3, Lossless=2).OnCUDA(0)

# (diff < (128 - thresh)) ? 20 : (diff > (128 + thresh)) ? 200 : 128
thresh = "5"
mt_makediff(my, ref, u=3, v=3).mt_lut("x 128 " + thresh + " - < 20 x 128 " + thresh + " + > 200 128 ? ?", u=3, v=3)

Interleave(last, ref.Subtitle("QTGMC", size=36), my.Subtitle("KTGMC", size=36))#.SelectEvery(6, 0, 1, 2)
DumpFilterGraph("T:\sandbox\t28\17\out.dot", mode=1)
DumpFilterGraph("T:\sandbox\t28\17\out.txt", mode=0)
return last
