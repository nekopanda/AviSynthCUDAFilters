### 補助インタレ解除(片フィールド補完) ###
[*
## 仕組み ##
1.Topフィールド、BottomフィールドのYUVplaneを比較し、差がfd_thresh未満の場合は何の処理も行いません。閾値以上の場合は各フィールド間のノイズ差分を計測しBobするか否かを決定します。(通常この設定を変える必要はありません。SDソースの場合は調整が必要です。)
2.まず両フィールドのノイズ平均を算出します。その値がnamax_thresh以上の場合は、ノイズ差分の閾値にnamax_diff + (off_tまたはoff_b)を用い、さらにノイズ量の多いフィールドを残します。これは映像効果としてのノイズがブロックノイズによって平坦化され、数値上のノイズ量が減少する場合への対処です。
   サンプル ttp://imgur.com/Cah6qsF,Zd0EeVQ,Dtt9vbJ,rWZUqtM,Hae0HLc,IhQ1WVq
3.namax_thresh未満の場合はy1-y5,x1-x5,off_t,off_bの数値によって決定される閾値よりも差分(の絶対値）が大きい場合に、綺麗なフィールドを用いてBobを行います。差分が小さい場合は何もしないか、nrtで定められる条件を満たした場合のみnrで指定したフィルタを掛けます。
閾値は以下のように決定します。x軸に綺麗なフィールドのノイズ量nmin、y軸に符号付ノイズ差分diff(top-bottom)をとり、(-∞,y1+off_b),(x1,y1+off_b)～(x5,y5+off_b),(∞,y5+off_b)の各点を結んだ線と(-∞,-y1-off_t),(x1,-y1-off_t)～(x5,-y5-off_t),(∞,-y5-off_t)を結んだ線で囲まれる範囲外がBoB対象。
   サンプル ttp://i.imgur.com/zXf2Lhd.png
尚、プリセットも用意しています。th_mode=1-5です。大きくなるほど検出率が上がり、検出漏れが増えます。x,yを調整する場合はmode0(デフォルト値はmode3と同じ)にして下さい。

4. off_t,off_bはフィールドに応じたノイズ差分の閾値のオフセットです。Topフィールドにより厳しい閾値を適用したい場合はoff_tを、Bottomならoff_bを上げてください。

## Debug ##
手動でBoBフレームを決定したい場合、debug=4or5(5は画像も出力)でtxt_fileで指定したファイルに候補フレームリストを抽出して手動修正した後、debug=6でそれらの設定を用いてエンコードを行えます。

## サンプル ##
[bf]ttp://i.imgur.com/nuFaUUz.png [af]ttp://i.imgur.com/ctiaIAV.png
[bf]ttp://i.imgur.com/bsHEoFw.png [af]ttp://i.imgur.com/8J9rSBN.png
[bf]ttp://i.imgur.com/YgQNeaA.png [af]ttp://i.imgur.com/mtjRjgM.png
*]

function KDecombUCF(clip s, float "y1",float "y2",float "y3",float "y4",float "y5",float "x1",float "x2",float "x3",float "x4",float "x5",float "off_t",float "off_b"
                                             \ ,int "th_mode", float "fd_thresh", float "namax_thresh",int "namax_diff"
                                             \ ,float "nrt1y",float "nrt2y",float "nrt2x",float "nrw",string "nr", int "chroma", int "debug", string "txt_file", int "frame_cache") {
## Decomb using clean field v3.13 ##
# needed dlls = Tdeint.dll, TMM.dll, NNEDI3.dll, TIVTC.dll, variableblur.dll, SmoothAdjust.dll, WarpSharp.dll
#正確な逆テレシネ後にも残ってしまう圧縮ノイズ由来の縞を、綺麗なフィールドを用いて補間・除去します。フィールドが共に汚い場合はBobではなく、任意のフィルタを掛けられます。
#補完対象フレームを増やしすぎると反ってファイルサイズ増えます(x264の動き予測に悪影響)。動きすぎて汚いフレームのみ推奨。
#未実装 : 高輝度は補完しない(高輝度ノイズは気付きにくい), 追加判定(輝度変化/field diff/FrameDiff),細かい模様が多いと共通部分が少なくなりnavgが急増・diff減少,nr機能向上
    [*    1st-stage analysis    *]
    fd_thresh = Default(fd_thresh, 128) #[0-] #threshold of FieldDiff #fd_thresh = FieldDiff * 100 / (Width * Height)
    [*    2nd-stage analysis    *]
    chroma = Default(chroma,1) #[0-2] #(0:Y),(1:UV),(2:YUV) for noise detection
    ## threshold
    th_mode = Default(th_mode, 0) #[1-2:debug][3-7:normal][8-10:restricted] #preset of diff threshold. you can also specify threshold by x1-x5 y1-y5(need th_mode=0).
    off_t = Default(off_t,0) #offset for diff threshold of top field (first field, top,diff<0)
    off_b = Default(off_b,0) #offset for diff threshold of bottom field (second field, botom, 0<diff)
    ## reverse (chroma=0のみで機能。ノイズ量の絶対値が多過ぎる場合、映像効果と考えノイズの大きいフィールドを残す(小さいほうはブロックノイズによる平坦化))
    namax_thresh = Default(namax_thresh, 82) #82 #MX:90 #[0-256] #disabled with chroma=1 #upper limit of max noise for Noise detaction (75-80-83)
    namax_diff = Default(namax_diff, 38) #30-40 #disabled with chroma=1  #If average noise >= namax_thresh,  use namax_diff as diff threshold.
    ## NR
    nrt1y = Default(nrt1y, 28)#28-29-30 #threshold for nr
    nrt2y = Default(nrt2y, 36)#36-36.5-37 #exclusion range
    nrt2x = Default(nrt2x, 53.5)#53-54-55 #exclusion range
    nrw = Default(nrw, 2)#1-2#diff weight for nr threshold
    DecombUCF_nr = Default(nr, """NOOP()""")#TNLMeans_SSE4_TNLMeans(Ax=5,Ay=5,Az=0,Sx=6,Sy=6,Bx=5,By=5,a=0.5,h=Spline(nmax ,0,2 ,28,2 ,38,6 ,100,7 ,false))""")
        #VerticalCleaner(mode=1,modeY=1,modeU=1,modeV=1).#閾値以上かつBOBが適用されないフレームに掛けるフィルタ(非常に汚いフレームのみ推奨)#NOOP()
    ## others
    nmin_y = 1         #[0-128] #lower limmit of luma noise for Noise detaction (1)
    nrange_y = 128   #[0-128] #luma range of luma noise for Noise detaction (128)
    nmin_uv = 1       #[0-128]
    nrange_uv = 128 #[0-128]
    debug = Default(debug,0) #[0-6] #(0:none),(1:Multi),(2:FieldDiff),(3:Show Noise),(4:Output frame list to txt in format for ConditionalReader),(5:Output frame list & stacked [C]fields as jpg),(6:Execute formal bob using external frame list),(7:Execute formal bob & output frame list)
    frame_cache = Default(frame_cache, 10) #逆テレシネ関数はフレーム再取得(第一、二判定、BOB)すると結果が変わる場合があるのでCache必須 #(3-5-10) #WriteFile(txt_file,"current_frame") 
    Assert(debug!=4&&debug!=5&&debug!=6&&debug!=7||Defined(txt_file), "If you use debug4,5,6,7 in DecombUCF(), you need to set txt_file.")
    txt_file = Default(txt_file, "nul")
    global chroma_y = (chroma==1) ? 2 : 3
    global chroma_uv = (chroma==0) ? 2 : 3
    c = s.FrameCache(frame_cache)
    ## diff threshold ## (x=nmin, y=diff)
    #                                                                    5      1     2     3     4    5     6     7     8    9    10  
    y1=Select(th_mode, Default(y1, 15), 13, 14, 15, 15, 15, 15, 15, 17, 20, 22)
    y2=Select(th_mode, Default(y2, 20), 17, 18, 19, 20, 20, 21, 22, 25, 32, 37)
    y3=Select(th_mode, Default(y3, 23), 17, 20, 21, 23, 23, 23, 24, 28, 38, 44)
    y4=Select(th_mode, Default(y4, 45), 20, 40, 43, 43, 45, 45, 45, 47, 52, 52)
    y5=Select(th_mode, Default(y5, 63), 50, 50, 63, 63, 63, 63, 63, 64, 66, 66)
    
    x1=Select(th_mode, Default(x1, 20), 20, 19, 20, 20, 20, 20, 20, 20, 21, 22)
    x2=Select(th_mode, Default(x2, 28), 28, 28, 28, 28, 28, 28, 28, 28, 30, 32)
    x3=Select(th_mode, Default(x3, 36), 32, 36, 36, 36, 36, 36, 35, 33, 36, 35)
    x4=Select(th_mode, Default(x4, 41), 37, 42, 41, 41, 41, 41, 41, 41, 40, 40)
    x5=Select(th_mode, Default(x5, 50), 50, 50, 53, 53, 50, 50, 50, 48, 48, 48)
    function th_score(float "x"){
        y =   (x<x1) ? y1
            \ :(x<x2) ? ((y2-y1)*x+x2*y1-x1*y2)/(x2-x1)
            \ :(x<x3) ? ((y3-y2)*x+x3*y2-x2*y3)/(x3-x2)
            \ :(x<x4) ? ((y4-y3)*x+x4*y3-x3*y4)/(x4-x3)
            \ :(x<x5) ? ((y5-y4)*x+x5*y4-x4*y5)/(x5-x4)
            \             : y5
        return y
    }
    function NOOP(clip c){c}
    function Eval_DecombUCF_nr(clip c, float "nmax"){c Eval(DecombUCF_nr)}
    ## Bob ##
    c
    #デバッグのため無効化
    #TDeint(chroma=true,mode=1,edeint=nnedi3(field=-2),emask=TMM(length=6,mode=1,nt=0,minthresh=0,maxthresh=75),map=0) #好きなBob関数(BOB関数によってはノイズ残る。comb-motion-noiseの3種類のマスクの方が良い) #判定結果に影響しません
    bob_t = SelectEven() #keep 1st field
    bob_b = SelectOdd() #keep 2nd field
    ## Noise detection ##
        # make diff # (検出したいノイズを特異的に変化させる)
        function Affect_Noise(clip c){
            c.binomialblur(3,3,chroma_y,chroma_uv,chroma_uv) #0.5,1.5が早い #好きなNRもしくはSharp。検出したいノイズに特異的に反応するものが良い。
        }
        function Limitter(clip c, int "nmin_y", int "nrange_y", int "nmin_uv", int "nrange_uv", int "chroma"){
            c
            range_y = nmin_y+nrange_y
            range_uv = nmin_uv+nrange_uv
            str_y = "(Y==128)? 128 : (Y<128)? ( ((127-"+String(range_y)+"<Y)(Y<128-"+String(nmin_y)+"))? 0 : 56 ) : ( ((128+"+String(nmin_y)+"<Y)(Y<129+"+String(range_y)+")) ? 255 : 199 )"
            str_u = "(U==128)? 128 : (U<128)? ( ((127-"+String(range_uv)+"<U)(U<128-"+String(nmin_uv)+"))? 0 : 56 ) : ( ((128+"+String(nmin_uv)+"<U)(U<129+"+String(range_uv)+")) ? 255 : 199)"
            str_v = "(V==128)? 128 : (V<128)? ( ((127-"+String(range_uv)+"<V)(V<128-"+String(nmin_uv)+"))? 0 : 56 ) : ( ((128+"+String(nmin_uv)+"<V)(V<129+"+String(range_uv)+")) ? 255 : 199)"
            eval_y = Select(chroma, str_y, "Y", str_y)
            eval_u = Select(chroma, "U", str_u, str_u)
            eval_v = Select(chroma, "V", str_v, str_v)
            return c.SmoothCustom(eval_y, eval_u, eval_v, false, 0, 0, -1)
        }
        c.SeparateFields()
        ori = Crop(4,4,-4,-4,true).BlankClip(color_yuv=$808080)
        ori_t = ori.SelectEven()
        ori_b = ori.SelectOdd()
        con = merge( Affect_Noise().invert() ).Crop(4,4,-4,-4,true) .Limitter(nmin_y,nrange_y,nmin_uv,nrange_uv,chroma).FrameCache(2)
        con_t = con.SelectEven()
        con_b = con.SelectOdd()
        # calc diff # (元フレームからの変化を絶対値で数値化)
            string1 = Select(chroma
                                       \,"""c = last
                            #絶対ノイズ量
                            noise_t_y = LumaDifference( ori_t, con_t )
                            noise_t_uv = -1
                            noise_b_y = LumaDifference( ori_b, con_b )
                            noise_b_uv = -1
                            #絶対ノイズ-平均(reverseで利用)
                            navg1_y =  ( noise_t_y + noise_b_y )/2 #Max(noise_t_y,noise_b_y)
                            #相対ノイズ-平均 [comp t,b](diff計算で利用)
                            navg2_y = LumaDifference( con_t, con_b )/2
                            #絶対ノイズ-符号付差分(diff計算で利用)
                            diff1_y = noise_t_y - noise_b_y
                            #パラメタ
                            diff1 = diff1_y #絶対ノイズ-符号付差分
                            navg1 = navg1_y #絶対ノイズ平均(総ノイズ量判定用)
                            navg1_d = navg1_y #debug用
                            navg2 = navg2_y #相対ノイズ-平均
                          """
                                       \,"""c = last
                            #絶対ノイズ量
                            noise_t_y = -1
                            noise_t_uv = ( ChromaUDifference( ori_t,con_t ) + ChromaVDifference( ori_t,con_t ) )/2
                            noise_b_y = -1
                            noise_b_uv = ( ChromaUDifference( ori_b,con_b ) + ChromaVDifference( ori_b,con_b ) )/2
                            #絶対ノイズ-平均(reverseで利用)
                            navg1_uv = ( noise_t_uv + noise_b_uv )/2 #Max(noise_t_uv,noise_b_uv)
                            #相対ノイズ-平均 [comp t,b](diff計算で利用)
                            navg2_uv = ( ChromaUDifference( con_t,con_b ) + ChromaVDifference( con_t,con_b ) )/4
                            #絶対ノイズ-符号付差分(diff計算で利用)
                            diff1_uv = noise_t_uv - noise_b_uv
                            #パラメタ
                            diff1 = diff1_uv #絶対ノイズ-符号付差分
                            navg1 = -1 #navg1_uv #絶対ノイズ平均(総ノイズ量判定用,色差の細かい模様は滅多に見ない)
                            navg1_d = navg1_uv #debug用
                            navg2 = navg2_uv #相対ノイズ-平均
                          """
                                       \,"""c = last
                            #絶対ノイズ量
                            noise_t_y = LumaDifference( ori_t,con_t )
                            noise_t_uv = ( ChromaUDifference( ori_t,con_t ) + ChromaVDifference( ori_t,con_t ) )/2
                            noise_b_y = LumaDifference( ori_b,con_b )
                            noise_b_uv = ( ChromaUDifference( ori_b,con_b ) + ChromaVDifference( ori_b,con_b ) )/2
                            #絶対ノイズ-平均(reverseで利用)
                            navg1_y =  ( noise_t_y + noise_b_y )/2 #Max(noise_t_y,noise_b_y)
                            navg1_uv = ( noise_t_uv + noise_b_uv )/2 #Max(noise_t_uv,noise_b_uv)
                            navg1_yuv = ( navg1_y + navg1_uv )/2
                            #相対ノイズ-平均 [comp t,b](diff計算で利用)
                            navg2_y = LumaDifference( con_t,con_b )/2
                            navg2_uv = ( ChromaUDifference( con_t,con_b ) + ChromaVDifference( con_t,con_b ) )/4
                            navg2_yuv = ( navg2_y + navg2_uv )/2
                            #絶対ノイズ-符号付差分(diff計算で利用)
                            diff1_y = noise_t_y - noise_b_y
                            diff1_uv = noise_t_uv - noise_b_uv
                            diff1_yuv = ( diff1_y + diff1_uv )/2
                            #パラメタ
                            diff1 = diff1_yuv #絶対ノイズ-符号付差分
                            navg1 = navg1_yuv #-1 #絶対ノイズ平均(総ノイズ量判定用,色差の細かい模様は滅多に見ない)
                            navg1_d = navg1_yuv #debug用
                            navg2 = navg2_yuv #相対ノイズ-平均
                          """
                                       \)

            string2 = """
                            nmin1 = navg2-Abs(diff1)/2
                            nmin = (nmin1<7) ? nmin1*4 : nmin1+21
                            nmax = navg2+Abs(diff1)*"""+String(nrw)+"""
                            off_thresh = (diff1<0)? """+String(off_t)+""" : """+String(off_b)+"""
                            min_thresh = (navg1<"""+String(namax_thresh)+""")? th_score(nmin) + off_thresh : """+String(namax_diff)+""" + off_thresh
                            #符号付補正差分
                            diff =    Abs(diff1)<1.8 ? diff1*10
                                   \ :Abs(diff1)<5   ? diff1*5 +(diff1/Abs(diff1))*9
                                   \ :Abs(diff1)<10  ? diff1*2 +(diff1/Abs(diff1))*24
                                   \                 : diff1   +(diff1/Abs(diff1))*34
                 """

            string3 = Select(debug
                                       \,"""
                            result = (Abs(diff) < min_thresh)                ? ((nmax<"""+String(nrt1y)+""")||("""+String(nrt2x)+"""<navg1_d && nmax<"""+String(nrt2y)+"""))
                                                                           \ ? c : c.Eval_DecombUCF_nr(nmax)
                                     \ : (navg1<"""+String(namax_thresh)+""") ? (diff<0) ? bob_t : bob_b
                                                                           \ : (diff<0) ? bob_b : bob_t
                            return result
                          """
                                       \,"""
                            debug1_n_t = Select("""+String(chroma)+""", " Noise  [Y : "+LeftStr(String(noise_t_y),8)+"]"
                                                                   \ , " Noise  [UV : "+LeftStr(String(noise_t_uv),8)+"]"
                                                                   \ , " Noise  [Y : "+LeftStr(String(noise_t_y),8)+"]  [UV : "+LeftStr(String(noise_t_uv),8)+"]" )
                            debug1_n_b = Select("""+String(chroma)+""", " Noise  [Y : "+LeftStr(String(noise_b_y),8)+"]"
                                                                   \ , " Noise  [UV : "+LeftStr(String(noise_b_uv),8)+"]"
                                                                   \ , " Noise  [Y : "+LeftStr(String(noise_b_y),8)+"]  [UV : "+LeftStr(String(noise_b_uv),8)+"]" )
                            debug1_o = c.Subtitle("[B] notbob  //  diff =  "+String(diff)+"  (|NoiseDiff| < "+String(min_thresh,"%.2f")+")")
                                      \ .Subtitle("Noise // Top "+MidStr(debug1_n_t,8)+" / Bottom "+MidStr(debug1_n_b,8),y=24)
                                      \ .Subtitle("navg1 : "+String(navg1_d,"%.2f")+" / nmin : "+String(nmin,"%.2f")+" / diff1 : "+String(diff1,"%.3f")+" / nrt : "+String(nmax,"%.1f") ,y=44)
                            debug1_t = StackHorizontal( bob_t.Subtitle(" [C] top        // diff : "+LeftStr(String(diff),8)+" ("
                                                                       \ +String(min_thresh,"%.2f")+") / Navg: "+String(navg1,"%.2f")+" ("""+String(namax_thresh)+""")",font_width=16)
                                                           \ .Subtitle(debug1_n_t,y=24,font_width=16)
                                                           \ .Subtitle(" navg1 : "+String(navg1_d,"%.2f")+" / nmin : "+String(nmin,"%.2f")+" / diff1 : "+String(diff1,"%.3f")+" / nrt : "+String(nmax,"%.1f") ,y=44,font_width=16)
                                                    \ , bob_b.Subtitle(" deleted",font_width=16).Subtitle(debug1_n_b,y=24,font_width=16) ).HorizontalReduceBy2()
                            debug1_b = StackHorizontal( bob_b.Subtitle(" [C] bottom // diff : "+LeftStr(String(diff),8)+" ("
                                                                       \ +String(min_thresh,"%.2f")+") / Navg: "+String(navg1,"%.2f")+" ("""+String(namax_thresh)+""")",font_width=16)
                                                           \ .Subtitle(debug1_n_b,y=24,font_width=16)
                                                           \ .Subtitle(" navg1 : "+String(navg1_d,"%.2f")+" / nmin : "+String(nmin,"%.2f")+" / diff1 : "+String(diff1,"%.3f")+" / nrt : "+String(nmax,"%.1f") ,y=44,font_width=16)
                                                    \ , bob_t.Subtitle(" deleted",font_width=16).Subtitle(debug1_n_t,y=24,font_width=16) ).HorizontalReduceBy2()
                            debug1 = (Abs(diff) < min_thresh)                ? ((nmax<"""+String(nrt1y)+""")||("""+String(nrt2x)+"""<navg1_d && nmax<"""+String(nrt2y)+"""))
                                                                           \ ? debug1_o : debug1_o.Eval_DecombUCF_nr(nmax).Subtitle("NR",y=62)
                                     \ : (navg1<"""+String(namax_thresh)+""") ? (diff<0) ? debug1_t : debug1_b 
                                                                           \ : (diff<0) ? debug1_b.Subtitle("Reversed",y=62) : debug1_t.Subtitle("Reversed",y=62)
                            return debug1
                          """
                                       \,"return c"
                                       \,"return c"
                                       \,"""
                            float1 = "%5.1f"
                            float4 = "%8.2f"
                            debug4_nr = c.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","3","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(diff/Abs(diff)*min_thresh,float1)+Chr(9)"
                                                                   \ ,"String(nmax,float1)+Chr(129)+Chr(161)+Chr(9)")
                            debug4_t = c.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","1","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(-min_thresh,float1)+Chr(9)")
                            debug4_b = c.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","2","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(min_thresh,float1)+Chr(9)")
                            debug4 = (Abs(diff) < min_thresh)                ? ((nmax<"""+String(nrt1y)+""")||("""+String(nrt2x)+"""<navg1_d && nmax<"""+String(nrt2y)+"""))
                                                                           \ ? c : debug4_nr
                                     \ : (navg1<"""+String(namax_thresh)+""") ? (diff<0) ? debug4_t : debug4_b 
                                                                           \ : (diff<0) ? debug4_b : debug4_t
                            return debug4
                          """
                                       \,"""
                            float1 = "%5.1f"
                            float4 = "%8.2f"
                            debug5_img = RevStr(MidStr(RevStr(""""+txt_file+""""),5)) + "_img_"
                            debug5_nr = c.Subtitle(String(current_frame)).WriteFile(""""+txt_file+"""","current_frame","Chr(9)","3","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)+Chr(9)+String(navg1_d,float1)+Chr(9)"
                                        \ ,"String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(diff/Abs(diff)*min_thresh,float1)+Chr(9)+String(nmax,float1)+Chr(129)+Chr(161)+Chr(9)")
                                        \.ConvertToRGB32(matrix="Rec709").ImageWriter(debug5_img,current_frame,-1,"jpg").ConvertToYV12(matrix="Rec709")
                            debug5_t = StackHorizontal(c.Bob().SelectEven(),c.Bob().SelectOdd()).HorizontalReduceBy2().Subtitle(String(current_frame))
                                       \ .WriteFile(""""+txt_file+"""","current_frame","Chr(9)","1","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)+Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)"
                                       \ ,"Chr(9)+String(AverageLuma,float1)+Chr(9)+String(-min_thresh,float1)+Chr(9)").ConvertToRGB32(matrix="Rec709").ImageWriter(debug5_img,current_frame,-1,"jpg").ConvertToYV12(matrix="Rec709")
                            debug5_b = StackHorizontal(c.Bob().SelectOdd(),c.Bob().SelectEven()).HorizontalReduceBy2().Subtitle(String(current_frame))
                                       \ .WriteFile(""""+txt_file+"""","current_frame","Chr(9)","2","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)+Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)"
                                       \ ,"Chr(9)+String(AverageLuma,float1)+Chr(9)+String(min_thresh,float1)+Chr(9)").ConvertToRGB32(matrix="Rec709").ImageWriter(debug5_img,current_frame,-1,"jpg").ConvertToYV12(matrix="Rec709")
                            debug5 = (Abs(diff) < min_thresh)                ? ((nmax<"""+String(nrt1y)+""")||("""+String(nrt2x)+"""<navg1_d && nmax<"""+String(nrt2y)+"""))
                                                                           \ ? c : debug5_nr 
                                     \ : (navg1<"""+String(namax_thresh)+""") ? (diff<0) ? debug5_t : debug5_b 
                                                                           \ : (diff<0) ? debug5_b : debug5_t
                            return debug5
                          """
                                       \,"return c"
                                       \,"""
                            float1 = "%5.1f"
                            float4 = "%8.2f"
                            debug7_nr = c.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","3","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(nmax,float1)+Chr(129)+Chr(161)+Chr(9)")
                            debug7_t = bob_t.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","1","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(-min_thresh,float1)+Chr(9)")
                            debug7_b = bob_b.WriteFile(""""+txt_file+"""","current_frame","Chr(9)","2","Chr(9)+Chr(35)+String(diff,float4)+Chr(9)+String(diff1,float1)"
                                                                   \ ,"Chr(9)+String(navg1_d,float1)+Chr(9)+String(nmin,float1)+Chr(9)+String(AverageLuma,float1)+Chr(9)+String(min_thresh,float1)+Chr(9)")
                            debug7 = (Abs(diff) < min_thresh)                ? ((nmax<"""+String(nrt1y)+""")||("""+String(nrt2x)+"""<navg1_d && nmax<"""+String(nrt2y)+"""))
                                                                           \ ? c : debug7_nr.Eval_DecombUCF_nr(nmax)
                                     \ : (navg1<"""+String(namax_thresh)+""") ? (diff<0) ? debug7_t : debug7_b 
                                                                           \ : (diff<0) ? debug7_b : debug7_t
                            return debug7
                          """
                                       \)
            # using FrameDiff (TIVTC)
    ## Apply to dirty frame ##
        # dmode1 : field diff detection #
        conditionalfilter(c
                            \, (debug==1)? c.ScriptClip("""subtitle("[A] notbob  //  Fdiff =  " + string(CFieldDiff(nt=0,chroma=true)*100.0/"""+String(Width(c)*Height(c))+""")+"  (FieldDiff < """+String(fd_thresh)+""")")""") : c
                            \, c.ScriptClip(string1+string2+string3)
                            \, "CFieldDiff(nt=0,chroma=true)*100.0/"+String(Width(c)*Height(c)),"<", String(0*fd_thresh) )
                            \.UseVar("y1", "y2", "y3", "y4", "y5", 
                            \             "x1", "x2", "x3", "x4", "x5")

        # addtional luma detection #第一判定に 輝度変化利用
    ## Debug ##
    #cleaned
    #(debug==1)? last.subtitle(" [A] : clean frame / [B] : noisy and both fields too (recommend only NR) / [C] : noisy but found clean field (BOB)",size=14,y=44,text_color=$00ff00) : NOP()
    (debug==2)? c.ScriptClip("""Subtitle(String(CFieldDiff(nt=0,chroma=true)*100.0/"""+String(Width(c)*Height(c))+""" ),y=40)""").FieldDiff(nt=0, chroma=true, display=true) : last
    (debug==3)? c.Crop(8,16,0,0,true).Loop(2).ScriptClip("""c = con
                             debug3_t = Interleave(c.SelectEven(),c.SelectEven())
                             debug3_b = Interleave(c.SelectOdd(),c.SelectOdd())
                             navg2_y = LumaDifference(debug3_t,debug3_b)/2
                             navg2_uv = ( ChromaUDifference(debug3_t,debug3_b) + ChromaVDifference(debug3_t,debug3_b) )/4
                             navg2_yuv = ( navg2_y+navg2_uv )/2
                             debug3_n2 = Select("""+String(chroma)+""", navg2_y, navg2_uv, navg2_yuv )
                             debug3_2 = Merge(debug3_t, debug3_b.Invert()).Subtitle("navg2 : "+String(debug3_n2,"%.2f"))
                             noise_y = LumaDifference(ori,c)
                             noise_uv = ( ChromaUDifference(ori,c) + ChromaVDifference(ori,c) )/2
                             debug3_n1 = Select("""+String(chroma)+""", " Noise  [Y : "+LeftStr(String( noise_y ),8)+"]"
                                                                  \ , " Noise  [UV : "+LeftStr(String( noise_uv ),8)+"]"
                                                                  \ , " Noise  [Y : "+LeftStr(String( noise_y ),8)+"]  [UV : "+LeftStr(String( noise_uv ),8)+"]" )
                             StackVertical(c,debug3_2).subtitle(debug3_n1,y=22)""").ScriptClip("""subtitle("frame = " + string(Float(current_frame)/2.0,"%.1f"))""")  : last
    (debug==4||debug==5||debug==7)? last.WriteFileStart(txt_file,""" "TYPE INT"+Chr(10)+"DEFAULT 0" """
                                        \ ,""" Chr(10)+"#flag:[0:NOP] [1:Bob 1st field] [2:Bob 2nd f.] [3:NR] [4:Copy bef. Frame] [5:Copy next F.] [6:Black16(TV)] [7:White235(TV)] [8:Black0(PC)] [9:White255(PC)]" """
                                        \ ,""" Chr(10)+"#"+Chr(34)+ScriptFile()+Chr(34) """
                                        \ ,""" Chr(10)+"#[chroma:"""+String(chroma)+"""] [nrt(nmax):"""+String(nrt1y,"%.1f")+"""]" ""","Chr(10)"
                                        \ ,""" Chr(10)+"#frame"+Chr(9)+"flag"+Chr(9)+"#diff"+Chr(9)+Chr(9)+"diff1"+Chr(9)+"navg1"+Chr(9)+"nmin"+Chr(9)+"luma"+Chr(9)+"diff_th"+Chr(9)+"nrt" """) : last
    (debug==6)? c.ScriptClip("""c = last
                             return Select(debug6_flag, c, bob_t, bob_b, c.Eval_DecombUCF_nr(), c, c
                                                    \ , c.BlankClip(color_yuv=$108080), c.BlankClip(color_yuv=$EB8080), c.BlankClip(color_yuv=$008080), c.BlankClip(color_yuv=$FF8080) ) """)
                   \    .ConditionalReader(txt_file,"debug6_flag").ScriptClip("""c = last
                             return Select(debug6_flag, c, c, c, c, c.Trim(current_frame-1,-1), c.Trim(current_frame+1,-1), c, c, c, c)""")
                   \    .ConditionalReader(txt_file,"debug6_flag") : last
    return last.UseVar("DecombUCF_nr",
                            \  "bob_t", "bob_b", "ori", "ori_t", "ori_b", "con", "con_t", "con_b")
}

LWLibavVideoSource("test.ts")
KDecombUCF()
