
#LoadPlugin("..\x64\Debug\InvertNegCUDA.dll")
#LoadPlugin("..\x64\Release\InvertNegCUDA.dll")

srcfile = "test.ts"
LWLibavVideoSource(srcfile).Time("Source")
Trim(356,356+300)

#OnCPU().Time("OnCPU")

#InvertNeg().Time("InvertNeg")
#InvertNeg().Time("InvertNeg2")

#OnCUDA().Time("OnCUDA")

