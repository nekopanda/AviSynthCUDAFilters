# KFMとKTGMCを使ったインタレース解除
# requires AvisynthNeo

function KFMDeint(clip src, int "mode", int "pass", string "preset", float "thswitch", bool "ucf", bool "nr", string "filepath", bool "cuda", int "dev", bool "show") {

    mode = default(mode, 0) # 0:VFR(60fps) 1:60fps 2:24fps
    pass = default(pass, 0) # 0:realtime 1:pre process(mode:0or2) 2:only timing(mode:0) 3:generate images
    preset = default(preset, "Faster") # QTGMC preset
    thswitch = default(thswitch, 0.5)
    ucf = default(ucf, false) # Enable DecombUCF
    nr = default(nr, false) # Enable Noise Reduction with (K)SMDegrain
    filepath = default(filepath, "cycle.dat")
    cuda = default(cuda, false) # Enable CUDA
    dev = default(dev, 0) # CUDA device index
    show = default(show, false) # Show information
    
    ivtc = Select(mode, true, false, true) # Enable Inverse Telecine
    out24 = Select(mode, false, false, true) # output 24fps
    
    Assert(!(mode == 1 && pass != 0), "60fps mode does not support multi-pass")
    Assert(!(mode == 2 && pass == 2), "24fps mode does not support timing output pass")

    # CUDAデバイスチェック
    if(cuda) {
        cc = CUDAComputeCapability(dev)
        Assert(cc >= 35, "[KFM CUDA Error] Specified GPU: " + CUDADeviceName(dev) + " is not usable. (CC=" + string(cc) + " < 35)")
    }
    
    OnDEV = cuda ? function[dev](clip c) { c.OnCUDA(2, dev) } : function(clip c) { c }

    # KFMPadは上下を4ピクセル（合計8ピクセル）拡張する
    pad = src.KFMPad()

    ## 60fpsクリップを作る ##

    # QTGMC/KTGMCを呼び出す useFlagは 0:通常 1:前のフィールドだけを使って補間 2:後ろのフィールドだけを使って補間
    # useFlagはKTGMCのみ対応（QTGMCは無視される）
    # 動きベクトルなどは再利用するので0を最初に呼び出してから1,2を呼び出すこと
    xTGMC = function[preset, cuda, dev](clip c, int useFlag) {
        prevGlobals = (useFlag == 0) ? "Replace" : "Reuse"
        cuda ? c.KTGMC(Preset=preset, dev=dev, PrevGlobals=prevGlobals, useFlag=useFlag) : c.QTGMC(Preset=preset).Prefetch(4)
    }

    # 16 == (Blocksize - Overlap): ブロックサイズに合わせる
    call_xTGMC = function[preset, src, pad, xTGMC](int useFlag) {
        is_pad = (preset == "Faster" && (Height(src) % 16) == 8)
        srcq = is_pad ? pad : src
        p60 = srcq.xTGMC(useFlag)
        is_pad ? p60.Crop(0,4,0,-4) : p60
    }
    
    # 静止部分補間用
    stt = src.KAnalyzeStatic(30, 15, pad)

    clip60 = call_xTGMC(0).KMergeStatic(src, stt)
    
    if(ucf) {
        # ノイズ情報クリップ作成関数
        UCFNoise = function[pad, cuda, OnDev](clip src) {
            if(cuda) {
                # ノイズ判定の準備。外周4ピクセルは判定から除外。Crop後にCUDAで処理するのにAlign必須
                fields = src.SeparateFields().Crop(4,4,-4,-4).Align()
                # オリジナルのDecombUCFだとCrop前にbinomialBlurを掛けていたが、CropとAlignが多くなるので
                # Crop後に掛けちゃう。多少結果が変わるかもしれないけどあまり変わらないと思う
                # オリジナルのbinomialBlurは理論的にはp=2.5相当（ただ見比べるとフィルタ範囲の違いよる差は結構出てる）
                noise = fields.KGaussResize(p=2.5)
                # ノイズ判定のための情報を集計。KAnalyzeNoiseの結果はCPUからアクセスされるのでOnCUDAしておく
                src.KAnalyzeNoise(fields.KNoiseClip(noise), pad).OnDEV()
            }
            else {
                fields = src.SeparateFields().Crop(4,4,-4,-4)
                w = fields.Width()
                h = fields.Height()
                epsilon = 0.0001
                noise = fields.GaussResize( w,h, 0,0, w+epsilon,h+epsilon, p=2.5 )
                src.KAnalyzeNoise(fields.KNoiseClip(noise), pad)
            }
        }

        # DecombUCFパラメータ
        ucfparam = KDecombUCFParam()
        # ノイズ情報クリップ
        noise30 = src.UCFNoise()
        
        if(cuda) {
            before60 = call_xTGMC(1).KMergeStatic(src, stt)
            after60 = call_xTGMC(2).KMergeStatic(src, stt)
        }
        else {
            # CPU版は対応していないのでTDeintを使う
            before60 = src.TDeint(chroma=true,mode=1,edeint=src.nnedi3(field=-2),emask=src.TMM2(mode=1, order=-1, field=-1))
            after60 = before60
        }
        
        if(show) {
            before60 = before60.Print("*** BOB BEFORE ***", y=20)
            after60 = after60.Print("*** BOB AFTER ***", y=20)
        }
    }
    
    if(ivtc) {

        if(show) {
            clip60 = clip60.Print("****** BOB ******", y=20)
        }

        ## 逆テレシネ解析 ##

        # KFM解析用クリップ
        super = src.KFMSuper(pad).KCleanSuper()

        # 30fpsのKFM解析用クリップ
        super30 = super.SelectEven()

        # テレシネ解析判定 KFMCycleAnalyzeはCUDA非対応なので注意
        mode = (pass >= 3) ? 2 : pass
        fmclip = super.KPreCycleAnalyze().OnDEV()
        fmclip = fmclip.KFMCycleAnalyze(src, mode=mode, filepath=filepath).OnCPU(0)
        
        if(pass == 1) {
            return fmclip
        }

        # 逆テレシネしたKFM解析用クリップ
        super24 = super.KTelecineSuper(fmclip)

        # 逆テレシネ 縞除去はノイズ低減のため
        #clip24 = src.KTelecine(fmclip).KFMPad().KRemoveCombe(super24)
        clip24 = pad.KTelecine(fmclip).KRemoveCombe(super24)

        if(ucf) {
            # DoubleWeaveしたクリップ 縞除去はノイズ低減のため
            #dweave = src.DoubleWeave().KFMPad().KRemoveCombe(super)
            dweave = pad.DoubleWeave().KRemoveCombe(super)

            if(show) {
                dweave = dweave.Print("** DOUBLE WEAVE **", y=20)
            }

            # DecombUCFを適用(24fps)
            clip24 = clip24.KDecombUCF24(ucfparam, fmclip, noise30, before60, after60, dweave)
        }

        if(nr) {
            if(cuda) {
                if(out24) {
                    # 24fps出力の時は高速化するため軽めの設定にする
                    clip24 = clip24.KSMDegrain(tr=1, chroma=False, binomial=True, dev=dev)
                }
                else {
                    clip24 = clip24.KSMDegrain(binomial=True, dev=dev)
                }
            }
            else {
                clip24 = clip24.SMDegrain(tr=1)
            }
        }

        # 24fps/60fps 切り替えマスク（maskとccをセットで使う）
        flag24 = super24.KSwitchFlag()
        mask24 = clip24.KCombeMask(flag24)
        cc24 = flag24.KContainsCombe().OnDEV() # CPUからアクセスされるのでOnCUDAしておく

        if(out24) {
            clip24.KPatchCombe(clip60, fmclip, mask24, cc24)
        }
        else {
            clip30 = show ? src.Print("****** SOURCE *****", y=20) : src
            if(ucf) {
                # DecombUCFを適用(30fps)
                clip30 = clip30.KDecombUCF(ucfparam, noise30, before60, after60)
            }
            if(nr) {
                # 30fpsは軽めの設定で行う
                if(cuda) {
                    clip30 = clip30.KSMDegrain(tr=1, chroma=False, binomial=True, dev=dev)
                }
                else {
                    clip30 = clip30.SMDegrain(tr=1)
                }
            }

            # 30fps/60fps 切り替えマスク（maskとccをセットで使う）
            flag30 = super30.KSwitchFlag()
            mask30 = clip30.KCombeMask(flag30)
            cc30 = flag30.KContainsCombe().OnDEV() # CPUからアクセスされるのでOnCUDAしておく

            # DecombUCFを適用(60fps)
            ucfc = ucf ? clip60.KDecombUCF60(ucfparam, noise30, before60, after60) : Undefined()

            mode = (pass == 3) ? 1 : pass
            result = clip60.KFMSwitch(fmclip, clip24, mask24, cc24, clip30, mask30, cc30, ucfc, thswitch=thswitch, mode=mode, show=show)

            result
        }
    }
    else {
        ucf ? clip60.KDecombUCF60(ucfparam, noise30, before60, after60) : clip60
    }
    
    if(cuda) {
        OnDEV()
    }
    else {
        Prefetch(4)
    }
}
