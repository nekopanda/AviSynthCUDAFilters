SetGraphAnalysis(true)
Import("MYBOB.avsi")
#Import("T:\sandbox\t28\14\Auto_Vfr.avs")
#LoadPlugin("X:\Switch\Visual Studio 2015\Projects\Amatsukaze\x64\Release\Amatsukaze.dll")
srcfile="T:\sandbox\t28\05\03_aikatsu.ts"
#AMTSource(srcfile)
LWLibavVideoSource(srcfile)
SMDegrain()
DumpFilterGraph("T:\sandbox\t28\14\graph_test.dot", mode=1)