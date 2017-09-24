
function QTGMC_BinomialSoften1(clip bobbed, bool chroma)
{
	# binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
	ts1 = bobbed.TemporalSoften( 1, 255,chroma ? 255 : 0, 28, 2 )  # 0.00  0.33  0.33  0.33  0.00
	return chroma ? ts1.Merge( bobbed, 0.25 ) : ts1.MergeLuma( bobbed, 0.25 )
}

function QTGMC_BinomialSoften2(clip bobbed, bool chroma)
{
	# binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
	ts1 = bobbed.TemporalSoften( 1, 255,chroma ? 255 : 0, 28, 2 )  # 0.00  0.33  0.33  0.33  0.00
	ts2 = bobbed.TemporalSoften( 2, 255,chroma ? 255 : 0, 28, 2 )  # 0.20  0.20  0.20  0.20  0.20
	return chroma ? ts1.Merge( ts2, 0.357 ).Merge( bobbed, 0.125 ) : ts1.Merge( ts2, 0.357 ).Merge( bobbed, 0.125 )
}
