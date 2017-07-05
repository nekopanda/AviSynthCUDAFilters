Import(".\MYBOB.avsi")
Import(".\KDEINT.avsi")

srcfile = "test.ts"
LWLibavVideoSource(srcfile)
AssumeTFF()

MYBOB().Time("MYBOB")

