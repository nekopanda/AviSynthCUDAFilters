
# Same as Bob, but keeps the field order the same.
function QTGMC_Bob(clip cp, float "b", float "c", int "height" )
{
  global debug1 = false
  h = cp.Height
  w = cp.Width
  shift = GetParity(cp) ? 0.25 : -0.25

  cp.SeparateFields()

  oeven=SelectEven()
  oodd=SelectOdd()

  even=oeven.ConvertToY().BicubicResize(w, h, b, c, 0, shift, w, Height()).AssumeFrameBased()
  odd=oodd.ConvertToY().BicubicResize(w, h, b, c, 0, -shift, w, Height()).AssumeFrameBased()

  evenChr=Interleave(oeven.ExtractU(), oeven.ExtractV()).BicubicResize(w/2, h/2, b, c, 0, shift, w/2, Height()/2).AssumeFrameBased()
  oddChr=Interleave(oodd.ExtractU(), oodd.ExtractV()).BicubicResize(w/2, h/2, b, c, 0, -shift, w/2, Height()/2).AssumeFrameBased()

  YToUV(Interleave(evenChr.SelectEven(), oddChr.SelectEven()),Interleave(evenChr.SelectOdd(), oddChr.SelectOdd()),Interleave(even, odd))
  GetParity(cp) ? AssumeTFF() : AssumeBFF()
}
