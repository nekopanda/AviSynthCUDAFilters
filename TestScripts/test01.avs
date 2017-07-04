
LoadPlugin("..\x64\Debug\InvertNegCUDA.dll")
#LoadPlugin("..\x64\Release\InvertNegCUDA.dll")

srcfile = "test.ts"
LWLibavVideoSource(srcfile)

OnCPU()

InvertNeg()
InvertNeg()

OnCUDA()
