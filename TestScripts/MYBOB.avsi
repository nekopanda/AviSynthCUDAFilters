

function MYBOB( clip Input, string "Preset", int "TR0", int "TR1", int "TR2", int "Rep0", int "Rep1", int "Rep2", string "EdiMode", bool "RepChroma", \
                    int "NNSize", int "NNeurons", int "EdiQual", int "EdiMaxD", string "ChromaEdi", int "EdiThreads", clip "EdiExt", float "Sharpness", \
                    int "SMode", int "SLMode", int "SLRad", int "SOvs", float "SVThin", int "Sbb", int "SrchClipPP", int "SubPel", int "SubPelInterp", \
                    int "BlockSize", int "Overlap", int "Search", int "SearchParam", int "PelSearch", bool "ChromaMotion", bool "TrueMotion", int "Lambda", \
                    int "LSAD", int "PNew", int "PLevel", bool "GlobalMotion", int "DCT", int "ThSAD1", int "ThSAD2", int "ThSCD1", int "ThSCD2", \
                    int "SourceMatch", string "MatchPreset", string "MatchEdi", string "MatchPreset2", string "MatchEdi2", int "MatchTR2", \
                    float "MatchEnhance", int "Lossless", int "NoiseProcess", float "EZDenoise", float "EZKeepGrain", string "NoisePreset", string "Denoiser", \
                    int "FftThreads", bool "DenoiseMC", int "NoiseTR", float "Sigma", bool "ChromaNoise", val "ShowNoise", float "GrainRestore", \
                    float "NoiseRestore", string "NoiseDeint", bool "StabilizeNoise", int "InputType", float "ProgSADMask", int "FPSDivisor", \
                    int "ShutterBlur", float "ShutterAngleSrc", float "ShutterAngleOut", int "SBlurLimit", bool "Border", bool "Precise", string "Tuning", \
                    bool "ShowSettings", string "GlobalNames", string "PrevGlobals", int "ForceTR", \
                    val "BT", val "DetailRestore", val "MotionBlur", val "MBlurLimit", val "NoiseBypass", float "Str", float "Amp", bool "TV_range", bool "useEdiExt", int "DftDither", bool "lsbd", bool "lsb" )
{
	# EdiMode="RepcYadif"/"cYadif" require the Yadif plugin, which doesn't autoload. Typically the calling script would load it.

	# Temporary Warnings
	Assert( !defined(BT),            "MYBOB: Setting BT has been replaced by setting NoiseTR" )
	Assert( !defined(DetailRestore), "MYBOB: Setting DetailRestore has been renamed to GrainRestore" )
	Assert( !defined(MotionBlur),    "MYBOB: Setting MotionBlur has been renamed to ShutterBlur" )
	Assert( !defined(MBlurLimit),    "MYBOB: Setting MBlurLimit has been renamed to SBlurLimit" )
	Assert( !defined(NoiseBypass),   "MYBOB: Setting NoiseBypass has been renamed to NoiseProcess" )

	#---------------------------------------
	# Presets
    lsbd         = default( lsbd,      false)
    lsb          = default( lsb,       false)
    useEdiExt    = default( useEdiExt, false)
    Str          = default( Str,        1.0 )
    Amp          = default( Amp,     0.0625 )
    TV_range     = default( TV_range,   !(Isrgb(input)))

        sispmt = Findstr(VersionString(), "AviSynth+") != 0 && Findstr(VersionString(), "r1576") == 0
	sispmt ? Assert( !(Input.BitsPerComponent() > 8 && (lsb || lsbd)), "MYBOB: lsb hack is not Compatible with native high bit depth" ) : nop()

	# Select presets / tuning
	Preset = default( Preset, "Slower" )
	pNum = (Preset == "Placebo"   ) ? 0 : \
	       (Preset == "Very Slow" ) ? 1 : \
	       (Preset == "Slower"    ) ? 2 : \
	       (Preset == "Slow"      ) ? 3 : \
	       (Preset == "Medium"    ) ? 4 : \
	       (Preset == "Fast"      ) ? 5 : \
	       (Preset == "Faster"    ) ? 6 : \
	       (Preset == "Very Fast" ) ? 7 : \
	       (Preset == "Super Fast") ? 8 : \
	       (Preset == "Ultra Fast") ? 9 : \
	       (Preset == "Draft"     ) ? 10 : 11
	Assert( pNum  < 11, "'Preset' choice is invalid" )

	mpNum1 = (!defined(MatchPreset))       ? ((pNum + 3 <= 9) ? (pNum + 3) : 9) : \
	         (MatchPreset == "Placebo"   ) ? 0 : \
	         (MatchPreset == "Very Slow" ) ? 1 : \
	         (MatchPreset == "Slower"    ) ? 2 : \
	         (MatchPreset == "Slow"      ) ? 3 : \
	         (MatchPreset == "Medium"    ) ? 4 : \
	         (MatchPreset == "Fast"      ) ? 5 : \
	         (MatchPreset == "Faster"    ) ? 6 : \
	         (MatchPreset == "Very Fast" ) ? 7 : \
	         (MatchPreset == "Super Fast") ? 8 : \
	         (MatchPreset == "Ultra Fast") ? 9 : \
	         (MatchPreset == "Draft"     ) ? 10 : 11
	Assert( mpNum1 < 10, "'MatchPreset' choice is invalid/unsupported" )
	MatchPreset = Select( mpNum1, "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast", "Draft" )

	mpNum2 = (!defined(MatchPreset2))       ? ((mpNum1 + 2 <= 9) ? (mpNum1 + 2) : 9) : \
	         (MatchPreset2 == "Placebo"   ) ? 0 : \
	         (MatchPreset2 == "Very Slow" ) ? 1 : \
	         (MatchPreset2 == "Slower"    ) ? 2 : \
	         (MatchPreset2 == "Slow"      ) ? 3 : \
	         (MatchPreset2 == "Medium"    ) ? 4 : \
	         (MatchPreset2 == "Fast"      ) ? 5 : \
	         (MatchPreset2 == "Faster"    ) ? 6 : \
	         (MatchPreset2 == "Very Fast" ) ? 7 : \
	         (MatchPreset2 == "Super Fast") ? 8 : \
	         (MatchPreset2 == "Ultra Fast") ? 9 : \
	         (MatchPreset2 == "Draft"     ) ? 10 : 11
	Assert( mpNum2 < 10, "'MatchPreset2' choice is invalid/unsupported" )
	MatchPreset2 = Select( mpNum2, "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast", "Draft" )

	NoisePreset = default( NoisePreset, "Fast" )
	npNum = (NoisePreset == "Slower" ) ? 0 : \
	        (NoisePreset == "Slow"   ) ? 1 : \
	        (NoisePreset == "Medium" ) ? 2 : \
	        (NoisePreset == "Fast"   ) ? 3 : \
	        (NoisePreset == "Faster" ) ? 4 : 5
	Assert( npNum < 5, "'NoisePreset' choice is invalid" )

	Tuning = default( Tuning, "None" )
	tNum = (Tuning == "None"  ) ? 0 : \
	       (Tuning == "DV-SD" ) ? 1 : \
	       (Tuning == "DV-HD" ) ? 2 : 3
	Assert( tNum < 3, "'Tuning' choice is invalid" )

	# Tunings only affect blocksize in this version
	bs = Select( tNum,  16, 16, 32 )
	bs2 = (bs >= 16) ? 32 : bs * 2

	#                                                               Very                                                        Very      Super      Ultra
	# Preset groups:                                     Placebo    Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast       Fast       Draft
	TR0          = default( TR0,          Select( pNum,  2,         2,        2,        2,        2,        2,        1,        1,        1,         1,         0      ) )
	TR1          = default( TR1,          Select( pNum,  2,         2,        2,        1,        1,        1,        1,        1,        1,         1,         1      ) )
	TR2X         = default( TR2,          Select( pNum,  3,         2,        1,        1,        1,        0,        0,        0,        0,         0,         0      ) )
	Rep0         = defined(Rep0) ? Rep0 : TR0 < 1 ? 0 : undefined()
	Rep0         = default( Rep0,         Select( pNum,  4,         4,        4,        4,        3,        3,        0,        0,        0,         0,         0      ) )
	Rep1         = default( Rep1,         Select( pNum,  0,         0,        0,        0,        0,        0,        0,        0,        0,         0,         0      ) )
	Rep2         = default( Rep2,         Select( pNum,  4,         4,        4,        4,        4,        4,        4,        4,        3,         3,         0      ) )
	EdiMode      = default( EdiMode,      Select( pNum, "NNEDI3",  "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3",  "RepYadif","Bob"   ) )
	NNSize       = default( NNSize,       Select( pNum,  1,         1,        1,        1,        5,        5,        4,        4,        4,         4,         4      ) )
	NNeurons     = default( NNeurons,     Select( pNum,  2,         2,        1,        1,        1,        0,        0,        0,        0,         0,         0      ) )
	EdiQual      = default( EdiQual,      Select( pNum,  1,         1,        1,        1,        1,        1,        1,        1,        1,         1,         1      ) )
	EdiMaxD      = default( EdiMaxD,      Select( pNum,  12,        10,       8,        7,        7,        6,        6,        5,        4,         4,         4      ) )
	SMode        = default( SMode,        Select( pNum,  2,         2,        2,        2,        2,        2,        2,        2,        2,         2,         0      ) )
	SLModeX      = default( SLMode,       Select( pNum,  2,         2,        2,        2,        2,        2,        2,        2,        0,         0,         0      ) )
	SLRad        = default( SLRad,        Select( pNum,  3,         1,        1,        1,        1,        1,        1,        1,        1,         1,         1      ) )
	Sbb          = default( Sbb,          Select( pNum,  3,         1,        1,        0,        0,        0,        0,        0,        0,         0,         0      ) )
	SrchClipPP   = default( SrchClipPP,   Select( pNum,  3,         3,        3,        3,        3,        2,        2,        2,        1,         1,         0      ) )
	SubPel       = default( SubPel,       Select( pNum,  2,         2,        2,        2,        1,        1,        1,        1,        1,         1,         1      ) )
	Blocksize    = default( Blocksize,    Select( pNum,  bs,        bs,       bs,       bs,       bs,       bs,       bs2,      bs2,      bs2,       bs2,       bs2    ) )
	bs = Blocksize
	Overlap      = default( Overlap,      Select( pNum,  bs/2,      bs/2,     bs/2,     bs/2,     bs/2,     bs/2,     bs/2,     bs/4,     bs/4,      bs/4,      bs/4   ) )
	Search       = default( Search,       Select( pNum,  5,         4,        4,        4,        4,        4,        4,        4,        0,         0,         0      ) )
	SearchParam  = default( SearchParam,  Select( pNum,  2,         2,        2,        2,        2,        2,        2,        1,        1,         1,         1      ) )
	PelSearch    = default( PelSearch,    Select( pNum,  2,         2,        2,        2,        1,        1,        1,        1,        1,         1,         1      ) )
	ChromaMotion = default( ChromaMotion, Select( pNum,  true,      true,     true,     false,    false,    false,    false,    false,    false,     false,     false  ) )
	Precise      = default( Precise,      Select( pNum,  true,      true,     false,    false,    false,    false,    false,    false,    false,     false,     false  ) )
	ProgSADMask  = default( ProgSADMask,  Select( pNum,  10.0,      10.0,     10.0,     10.0,     10.0,     0.0,      0.0,      0.0,      0.0,       0.0,       0.0    ) )

	# Noise presets                                           Slower     Slow       Medium     Fast       Faster
	Denoiser       = default( Denoiser,       Select( npNum, "dfttest", "dfttest", "dfttest", "fft3df",  "fft3df" ) )
	DenoiseMC      = default( DenoiseMC,      Select( npNum,  true,      true,      false,     false,     false   ) )
	NoiseTR        = default( NoiseTR,        Select( npNum,  2,         1,         1,         1,         0       ) )
	NoiseDeint     = default( NoiseDeint,     Select( npNum, "Generate","Bob",      "",        "",        ""      ) )
	StabilizeNoise = default( StabilizeNoise, Select( npNum,  true,      true,      true,      false,     false   ) )

	# The basic source-match step corrects and re-runs the interpolation of the input clip. So it initialy uses same interpolation settings as the main preset
	SourceMatch   = default( SourceMatch, 0 )
	MatchNNSize   = NNSize
	MatchNNeurons = NNeurons
	MatchEdiMaxD  = EdiMaxD
	MatchEdiQual  = EdiQual

	# However, can use a faster initial interpolation when using source-match allowing the basic source-match step to "correct" it with higher quality settings
	Assert( SourceMatch == 0 || mpNum1 >= pNum, "'MatchPreset' cannot use a slower setting than 'Preset'" )
	#                                                                    Very                                                        Very      Super     Ultra
	# Basic source-match presets                                Placebo  Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast      Fast
	NNSize   = (SourceMatch == 0) ? NNSize   : Select( mpNum1,  1,       1,        1,        1,        5,        5,        4,        4,        4,        4     )
	NNeurons = (SourceMatch == 0) ? NNeurons : Select( mpNum1,  2,       2,        1,        1,        1,        0,        0,        0,        0,        0     )
	EdiMaxD  = (SourceMatch == 0) ? EdiMaxD  : Select( mpNum1,  12,      10,       8,        7,        7,        6,        6,        5,        4,        4     )
	EdiQual  = (SourceMatch == 0) ? EdiQual  : Select( mpNum1,  1,       1,        1,        1,        1,        1,        1,        1,        1,        1     )
	TempEdi  = EdiMode # Main interpolation is actually done by basic-source match step when enabled, so a little swap and wriggle is needed
	EdiMode  = (SourceMatch == 0) ? EdiMode  : default( MatchEdi, ((mpNum1 < 9) ?  EdiMode : "Yadif") ) # Force Yadif for "Ultra Fast" basic source match
	MatchEdi = TempEdi

	#                                                          Very                                                        Very      Super     Ultra
	# Refined source-match presets                   Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast      Fast
	MatchEdi2 = default( MatchEdi2, Select( mpNum2, "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "NNEDI3", "TDeint",  ""    ) )
	MatchNNSize2                  = Select( mpNum2,  1,        1,        1,        1,        5,        5,        4,        4,        4,        4     )
	MatchNNeurons2                = Select( mpNum2,  2,        2,        1,        1,        1,        0,        0,        0,        0,        0     )
	MatchEdiMaxD2                 = Select( mpNum2,  12,       10,       8,        7,        7,        6,        6,        5,        4,        4     )
	MatchEdiQual2                 = Select( mpNum2,  1,        1,        1,        1,        1,        1,        1,        1,        1,        1     )


	#---------------------------------------
	# Settings

	# Core and Interpolation defaults
	TR2        = (SourceMatch > 0) ? default(TR2, ((TR2X == 0) ? 1 : TR2X)) : TR2X  # ***TR2 defaults always at least 1 when using source-match***
	RepChroma  = default( RepChroma,  true )
	EdiThreads = default( EdiThreads, 0    )
	ChromaEdi  = default( ChromaEdi,  ""   )
	NNeurons   = (EdiMode == "NNEDI2" && NNeurons > 2) ? 2 : NNeurons # Smaller range for NNeurons in NNEDI2 (which calls it nsize)
	EdiQual    = (EdiMode == "EEDI3+NNEDI3" || EdiMode == "NNEDI3") && EdiQual > 2 ? 2 : EdiQual  # Smaller range for EdiQual in NNEDI3

	# Source-match / lossless defaults
	MatchTR1     = TR1
	MatchTR2     = default( MatchTR2,     1   )
	MatchEnhance = default( MatchEnhance, 0.5 )
	Lossless     = default( Lossless,     0   )
	Assert( Lossless <= 2, "Lossless setting only supports mode 1 ('true lossless') and mode 2 ('fake lossless') - see documentation in script and consider source-match settings" )

	# Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match), but adjusted to give roughly same sharpness for all settings
	SMode      = (defined(Sharpness) && Sharpness == 0.0) ? 0 : SMode
	SLMode     = (SourceMatch > 0) ? default(SLMode, 0) : SLModeX  # ***Sharpness limiting disabled by default for source-match***
	SLMode     = (SLRad <= 0)      ? 0 : SLMode
	spatialSL  = (SLMode == 1 || SLMode == 3)
	temporalSL = (SLMode == 2 || SLMode == 4)
	Sharpness  = default( Sharpness, (SMode == 0) ? 0.0 : ((SourceMatch > 0) ? 0.2 : 1.0) )      # Default sharpness is 1.0, or 0.2 if using source-match
	sharpMul   = (temporalSL) ? 2 : (spatialSL) ? 1.5 : 1                                        # Adjust sharpness based on other settings
	sharpAdj   = Sharpness * (sharpMul * (0.2 + TR1*0.15 + TR2*0.25) + ((SMode == 1) ? 0.1 : 0)) # [This needs a bit more refinement]
	Sbb        = (SMode == 0) ? 0 : Sbb
	SOvs       = default( SOvs,   0   )
	SVThin     = default( SVThin, 0.0 )

	# Noise processing settings
	Assert( !defined(EZDenoise) || EZDenoise <= 0.0 || !defined(EZKeepGrain) || EZKeepGrain <= 0.0, "MYBOB: EZDenoise and EZKeepGrain cannot be used together" )
	NoiseProcess = defined(NoiseProcess) ? NoiseProcess : \
	               (defined(EZDenoise)   && EZDenoise   > 0.0)    ? 1 : \
	               (defined(EZKeepGrain) && EZKeepGrain > 0.0)    ? 2 : \
				   (Preset == "Placebo" || Preset == "Very Slow") ? 2 : 0
	GrainRestore = defined(GrainRestore) ? GrainRestore : \
	               (defined(EZDenoise)   && EZDenoise   > 0.0) ? 0.0 : \
	               (defined(EZKeepGrain) && EZKeepGrain > 0.0) ? 0.3 * sqrt(EZKeepGrain) : \
				                                                Select( NoiseProcess, 0.0, 0.7, 0.3 )
	NoiseRestore = defined(NoiseRestore) ? NoiseRestore : \
	               (defined(EZDenoise)   && EZDenoise   > 0.0) ? 0.0 : \
	               (defined(EZKeepGrain) && EZKeepGrain > 0.0) ? 0.1 * sqrt(EZKeepGrain) : \
				                                                Select( NoiseProcess, 0.0, 0.3, 0.1 )
	Sigma        = defined(Sigma)       ? Sigma : \
	               (defined(EZDenoise)   && EZDenoise   > 0.0) ? EZDenoise : \
	               (defined(EZKeepGrain) && EZKeepGrain > 0.0) ? 4.0 * EZKeepGrain : 2.0
	FftThreads   = default( FftThreads, EdiThreads )
	ChromaNoise  = default( ChromaNoise, false )
	ShowNoise    = default( ShowNoise, 0.0 )
	ShowNoise    = IsBool( ShowNoise ) ? (ShowNoise ? 10.0 : 0.0) : ShowNoise
	NoiseProcess = (ShowNoise > 0.0)   ? 2   : NoiseProcess
	NoiseRestore = (ShowNoise > 0.0)   ? 1.0 : NoiseRestore
	NoiseTR      = (NoiseProcess == 0) ? 0   : NoiseTR
	GrainRestore = (NoiseProcess == 0) ? 0.0 : GrainRestore
	NoiseRestore = (NoiseProcess == 0) ? 0.0 : NoiseRestore
	totalRestore = GrainRestore + NoiseRestore
	StabilizeNoise = (totalRestore <= 0) ? false : StabilizeNoise
	noiseTD  = Select( NoiseTR, 1, 3, 5 )
	noiseCentre = (Denoiser == "dfttest") ? "128" : "128.5"

	# MVTools settings
	TrueMotion   = default( TrueMotion,   false )
	GlobalMotion = default( GlobalMotion, true  )
	Lambda       = default( Lambda, ((TrueMotion) ? 1000 : 100 ) * (BlockSize*BlockSize)/(8*8) )
	LSAD         = default( LSAD,    (TrueMotion) ? 1200 : 400 )
	PNew         = default( PNew,    (TrueMotion) ? 50   : 25  )
	PLevel       = default( PLevel,  (TrueMotion) ? 1    : 0   )
	DCT          = default( DCT,     0          )
	ThSAD1       = default( ThSAD1,  10 * 8*8   )
	ThSAD2       = default( ThSAD2,   4 * 8*8   )
	ThSCD1       = default( ThSCD1,  180        )
	ThSCD2       = default( ThSCD2,  98         )

	#---------------------------------------
	# Pre-Processing

	w = Input.Width()
	h = Input.Height()
	epsilon = 0.0001

	clip = Input

	#---------------------------------------
	# Motion Analysis

	# Bob the input as a starting point for motion search clip
	bobbed = clip.Time("Input").MYBOB_Bob( 0,0.5 ).Time("Bob")

	# binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
	ts1 = bobbed.TemporalSoften( 1, 255,255, 28, 2 )  # 0.00  0.33  0.33  0.33  0.00
	ts2 = bobbed.TemporalSoften( 2, 255,255, 28, 2 )  # 0.20  0.20  0.20  0.20  0.20
	binomial0 = ts1.Merge( ts2, 0.357 ).Merge( bobbed, 0.125 ).Time("binomial0")

	# Remove areas of difference between temporal blurred motion search clip and bob that are not due to bob-shimmer - removes general motion blur
	repair0 = binomial0.MYBOB_KeepOnlyBobShimmerFixes( bobbed ).Time("repair0")

	# Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
	# a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
	spatialBlur = repair0.RemoveGrain( 12 ).GaussResize( w,h, 0,0, w+epsilon,h+epsilon, p=2 )

	spatialBlur = spatialBlur.Merge( repair0, 0.1 ).Time("spatialBlur")
	tweaked     = mt_lutxy( repair0, bobbed, "x 3 scalef + y < x 3 scalef + x 3 scalef - y > x 3 scalef - y ? ?", U=3,V=3 )
	srchClip    = spatialBlur.mt_lutxy( tweaked, "x 7 scalef + y < x 2 scalef + x 7 scalef - y > x 2 scalef - x 51 * y 49 * + 100 / ? ?", U=3,V=3 )

	# tv range -> full range
	srchClip    = srchClip.mt_lut(yexpr="x 16 - 219 / 0 1 clip 255 *", expr="x 128 - 128 * 112 / 128 +", U=3,V=3).Time("srchClip")

	# Calculate forward and backward motion vectors from motion search clip
	srchSuper = srchClip.MSuper().Time("srchSuper")
	
	bVec2 = srchSuper.MAnalyse( isb=true,  delta=2, blksize=16, overlap=8, search=4, searchparam=2, \
                pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0, chroma=true ).Time("bVec2")
	bVec1 = srchSuper.MAnalyse( isb=true,  delta=1, blksize=16, overlap=8, search=4, searchparam=2, \
                pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0, chroma=true ).Time("bVec1")
	fVec1 = srchSuper.MAnalyse( isb=false, delta=1, blksize=16, overlap=8, search=4, searchparam=2, \
	            pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0, chroma=true ).Time("fVec1")
	fVec2 = srchSuper.MAnalyse( isb=false, delta=2, blksize=16, overlap=8, search=4, searchparam=2, \
                pelsearch=2, truemotion=false, lambda=400, lsad=400, pnew=25, plevel=0, global=true, DCT=0, chroma=true ).Time("fVec2")

	#---------------------------------------
	# Noise Processing

	#---------------------------------------
	# Interpolation

	# Create interpolated image as starting point for output
	edi = clip.NNEDI3( field=-2, nsize=NNSize, nns=NNeurons, threads=EdiThreads ).Time("NNEDI3")

	# Get the max/min value for each pixel over neighboring motion-compensated frames - used for temporal sharpness limiting
	ediSuper = edi.MSuper().Time("ediSuper")
	bComp1   = edi.MCompensate( ediSuper, bVec1, thSCD1=ThSCD1,thSCD2=ThSCD2              ).Time("bComp1")
	fComp1   = edi.MCompensate( ediSuper, fVec1, thSCD1=ThSCD1,thSCD2=ThSCD2              ).Time("fComp1")
	tMax     = edi.mt_logic( fComp1, "max", U=3,V=3 ).mt_logic( bComp1, "max", U=3,V=3 )
	tMin     = edi.mt_logic( fComp1, "min", U=3,V=3 ).mt_logic( bComp1, "min", U=3,V=3 )

	#---------------------------------------
	# Create basic output

	# Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames. As above, this is done to remove shimmer from
	# alternate frames so the same binomial kernels are used. However, by using motion-compensated smoothing this time we avoid motion blur. The use of
	# MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different, but they evaluate to the same values
	# Create linear weightings of neighbors first                                                                                -2    -1     0    1     2
	degrain1 = edi.MDegrain1( ediSuper, bVec1,fVec1, thSAD=ThSAD1, thSCD1=ThSCD1,thSCD2=ThSCD2, lsb=lsb)  # 0.00  0.33  0.33  0.33  0.00
	degrain2 = edi.MDegrain1( ediSuper, bVec2,fVec2, thSAD=ThSAD1, thSCD1=ThSCD1,thSCD2=ThSCD2, lsb=lsb)  # 0.33  0.00  0.33  0.00  0.33

	# Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
	lossed1 = degrain1.Merge( degrain2, 0.2 ).Merge( edi, 0.0625 ).Time("lossed1")

	#---------------------------------------
	# Resharpen / retouch output

	# Resharpen to counteract temporal blurs. Little sharpening needed for source-match mode since it has already recovered sharpness from source
	vresharp = Merge( lossed1.mt_expand( mode="vertical", U=3,V=3 ), lossed1.mt_inpand( mode="vertical", U=3,V=3 ) )
	thin   = lossed1.mt_lutxy( vresharp.RemoveGrain( 12 ), "clamp_f x x y - "+ string(sharpAdj) + " * +", U=3,V=3 )

	# Back blend the blurred difference between sharpened & unsharpened clip, before (1st) sharpness limiting (Sbb == 1,3). A small fidelity improvement
	backBlend1 = thin.mt_makediff( mt_makediff( thin, lossed1, U=1,V=1 ).RemoveGrain( 12, -1 ) \
	                         .GaussResize( w,h, 0,0, w+epsilon,h+epsilon, p=5 ), U=2,V=2 )

	# Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
	# Occurs here (before final temporal smooth) if SLMode == 1,2. This location will restrict sharpness more, but any artefacts introduced will be smoothed
	addNoise1 = backBlend1.mt_clamp( tMax,tMin, Sovs,Sovs, U=3,V=3 ).Time("addNoise1")

	
	# Final light linear temporal smooth for denoising
	stableSuper = addNoise1.MSuper()
	stable  = addNoise1.MDegrain1( stableSuper, bVec1,fVec1,thSAD=ThSAD2, thSCD1=ThSCD1,thSCD2=ThSCD2, lsb=lsb ).Time("stable")

	# Remove areas of difference between final output & basic interpolated image that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
	repair2 = stable.MYBOB_KeepOnlyBobShimmerFixes( edi ).Time("repair2")

	return repair2
}


#---------------------------------------
# Helpers

# Same as Bob, but keeps the field order the same.
function MYBOB_Bob(clip cp, float "b", float "c", int "height" )
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

# Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference, i.e. bob shimmer fixes
# Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed
#                  by this process. Restore remaining areas of difference back to as they were in reference clip.
function MYBOB_KeepOnlyBobShimmerFixes( clip Input, clip Ref )
{
	diff = mt_makediff(Ref, Input, U=3,V=3 )

	# Areas of positive difference
	choke1 = diff.  mt_inpand( mode="vertical", U=3,V=3 )
	choke1 = choke1.mt_inpand( mode="vertical", U=3,V=3 )
	choke1 = choke1.mt_logic( choke1.RemoveGrain( 20 ), "min", Y=3,U=3,V=3 )
	choke1 = choke1.mt_expand( mode="vertical", U=3,V=3 )
	choke1 = choke1.mt_expand( mode="vertical", U=3,V=3 )

	# Areas of negative difference (similar to above)
	choke2 = diff.  mt_expand( mode="vertical", U=3,V=3 )
	choke2 = choke2.mt_expand( mode="vertical", U=3,V=3 )
	choke2 = choke2.mt_logic( choke2.RemoveGrain( 20 ), "max", Y=3,U=3,V=3 )
	choke2 = choke2.mt_inpand( mode="vertical", U=3,V=3 )
	choke2 = choke2.mt_inpand( mode="vertical", U=3,V=3 )

	# Combine above areas to find those areas of difference to restore
	restore = diff.mt_lutxy( choke1, "x 129 scalef < x y range_half < range_half y ? ?", U=3,V=3 ).mt_lutxy( choke2, "x 127 scalef > x y range_half > range_half y ? ?", U=3,V=3 )

	return Input.mt_adddiff( restore, U=3,V=3 )
}
