Import(".\MYBOB.avsi")
Import(".\KDEINT.avsi")

srcpath = "test.ts"
refpath="qtgmc_ref.y4m"

orig = LWLibavVideoSource(srcpath).Trim(356,356+300-1)


my = orig.MYBOB()

ref = LWLibavVideoSource(refpath)

#diff = mt_makediff(my, ref)
diff = mt_makediff(my, ref, u=3, v=3)
# (diff < 128) ? 20 : (diff > 128) ? 200 : 128
diff.mt_lut("x 128 < 20 diff 128 > 200 128 ? ?", u=3, v=3)

return Interleave(last, ref.Subtitle("QTGMC", size=36), my.Subtitle("MYBOB", size=36)).SelectEvery(6, 0, 1, 2)

Subtitle(String(debug1))
