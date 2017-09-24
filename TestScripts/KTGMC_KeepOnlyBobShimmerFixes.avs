function KTGMC_KeepOnlyBobShimmerFixes( clip Input, clip Ref, int Rep, bool Chroma )
{
	ed = Rep
	RCmt = Chroma ? 3   :  1
	RCmto = Chroma ? 3 : 2
	RCrg = Chroma ? 20  : -1

	diff = KMakeDiff(Ref, Input, U=RCmt,V=RCmt )

	# Areas of positive difference
	choke1 =                        diff.  KInpandVerticalX2( U=RCmt,V=RCmt )
	choke1 = (ed % 3 != 0)        ? choke1.KLogic( choke1.KRemoveGrain( 20, RCrg ), "min", U=RCmt,V=RCmt ) : choke1
	choke1 =                        choke1.KExpandVerticalX2( U=RCmt,V=RCmt )

	# Areas of negative difference (similar to above)
	choke2 =                        diff.  KExpandVerticalX2( U=RCmt,V=RCmt )
	choke2 = (ed % 3 != 0)        ? choke2.KLogic( choke2.KRemoveGrain( 20, RCrg ), "max", U=RCmt,V=RCmt ) : choke2
	choke2 =                        choke2.KInpandVerticalX2( U=RCmt,V=RCmt )

	# Combine above areas to find those areas of difference to restore
	return Input.KTGMC_BobShimmerFixesMerge(diff, choke1, choke2, U=RCmto, V=RCmto)
}
