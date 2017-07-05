Import(".\MYBOB.avsi")
Import(".\KDEINT.avsi")

srcfile = "test.ts"
LWLibavVideoSource(srcfile)
Trim(356,356+300-1)
AssumeTFF()

QTGMC()

