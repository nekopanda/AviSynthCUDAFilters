LWLibavVideoSource("test.ts")
### Clean scene change ###
function Clean_SC(clip clip, float "b_diff_th", float "b_dup_th", float "d_diff_th", float "d_dup_th", float "luma_boundary", int "show_mode"){
  b_diff_th = default(b_diff_th, 20.0) 
  b_dup_th = default(b_dup_th, 5.0)#mx,sun=10,その他=5.0
  d_diff_th = default(d_diff_th, 8.0)
  d_dup_th = default(d_dup_th, 1.6)#mx,sun=2,その他=1.6
  luma_boundary = default(luma_boundary, 48.0)
  show_mode = default(show_mode, 0)
  function Clean_SC_main(clip clip, float "diff_th", float "dup_th"){
    clip
    copy = Dup1(threshold=dup_th, copy=true, maxcopies=1)
    ConditionalFilter(last, copy, last, "YDifferenceFromPrevious()", ">", String(diff_th))
    Reverse()
    copy = Dup1(threshold=dup_th, copy=true, maxcopies=1)
    ConditionalFilter(last, copy, last, "YDifferenceFromPrevious()", ">", String(diff_th))
    Reverse()
  }
  clip
  ConditionalFilter(last, Clean_SC_main(d_diff_th,d_dup_th), Clean_SC_main(b_diff_th,b_dup_th), "AverageLuma()", "<", String(luma_boundary))
  return (show_mode == 0) ? last : clip.Dup(copy=false,show=true).ScriptClip("""subtitle("Luma = "+string(AverageLuma())+"  /Diff = "+string(YDifferenceToNext()))""")
}
function KClean_SC(clip clip, float "b_diff_th", float "b_dup_th", float "d_diff_th", float "d_dup_th", float "luma_boundary", int "show_mode"){
  b_diff_th = default(b_diff_th, 20.0) 
  b_dup_th = default(b_dup_th, 5.0)#mx,sun=10,その他=5.0
  d_diff_th = default(d_diff_th, 8.0)
  d_dup_th = default(d_dup_th, 1.6)#mx,sun=2,その他=1.6
  luma_boundary = default(luma_boundary, 48.0)
  show_mode = default(show_mode, 0)
  
  clip
  prev = DuplicateFrame(0)
  next = DeleteFrame(0)
  diff = AddProp("val", "KCFrameDiffDup()")
  ScriptClip("""
  
  # 平均輝度から閾値を決定
  is_low_luma = (KAverageLuma() < """+string(luma_boundary)+""")
  diff_th = is_low_luma ? """+string(d_diff_th)+""" : """+string(b_diff_th)+"""
  dup_th = is_low_luma ? """+string(d_dup_th)+""" : """+string(b_dup_th)+"""
  
  # 判定
  is_copy_prev = (KYDifferenceToNext() > diff_th) && (diff.GetProp("val", 0) < dup_th)
  is_copy_next = (KYDifferenceFromPrevious() > diff_th) && (diff.GetProp("val", 1) < dup_th)
  
  # 判定結果を返す
  is_copy_prev ? prev : is_copy_next ? next : last
  
  """).UseVar("prev", "next", "diff")
  
  return (show_mode == 0) ? last : clip.ScriptClip("""
  subtitle("Luma = "+string(KAverageLuma())+"  /Diff = "+string(KYDifferenceToNext())+"   /Dup = "+string(diff.GetProp("val", 1)))
  """).UseVar("diff")
}
#Clean_SC()
OnCPU(8).KClean_SC().OnCUDA(8)
#mt_lutxy(ref, cuda, mt_polish("((x - y) == 0) ? 128 : 0"),U=3,V=3)
#StackHorizontal(last,ref,cuda)
#Interleave(ref,cuda)