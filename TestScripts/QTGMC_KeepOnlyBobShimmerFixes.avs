# Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed
#                  by this process. Restore remaining areas of difference back to as they were in reference clip.
function QTGMC_KeepOnlyBobShimmerFixes( clip Input, clip Ref, int Rep, bool Chroma )
{
	# >>>> YUY2 is planar here

	# ed is the erosion distance - how much to deflate then reflate to remove thin areas of interest: 0 = minimum to 6 = maximum
	# od is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:  0 = none to 3 = one full pixel
	# If Rep < 10, then ed = Rep and od = 0, otherwise ed = 10s digit and od = 1s digit (nasty method, but kept for compatibility with original TGMC)
	Rep    = default( Rep,    1    )
	Chroma = default( Chroma, true )
	ed = (Rep < 10) ? Rep : Rep / 10
	od = (Rep < 10) ? 0   : Rep % 10
	RCrg  = Chroma ? 3 : 1
	RCrgo = Chroma ? 3 : 2

	diff = mt_makediff(Ref, Input, U=3,V=3 )

	# Areas of positive difference                                                                # ed = 0 1 2 3 4 5 6 7
	choke1 =                        diff.  mt_inpand( mode="vertical", U=RCrg,V=RCrg )            #      x x x x x x x x    1 pixel   \
	choke1 = (ed > 2)             ? choke1.mt_inpand( mode="vertical", U=RCrg,V=RCrg ) : choke1   #      . . . x x x x x    1 pixel    |  Deflate to remove thin areas
	choke1 = (ed > 5)             ? choke1.mt_inpand( mode="vertical", U=RCrg,V=RCrg ) : choke1   #      . . . . . . x x    1 pixel   /
	choke1 = (ed % 3 != 0)        ? choke1.QTGMC_deflate( U=RCrg,V=RCrg )              : choke1   #      . x x . x x . x    A bit more deflate & some horizonal effect
	choke1 = (ed == 2 || ed == 5) ? choke1.RemoveGrain( 4              )               : choke1   #      . . x . . x . .    Local median
                                                                                                                      
	choke1 =                        choke1.mt_expand( mode="vertical", U=RCrg,V=RCrg )            #      x x x x x x x x    1 pixel  \
	choke1 = (ed > 1)             ? choke1.mt_expand( mode="vertical", U=RCrg,V=RCrg ) : choke1   #      . . x x x x x x    1 pixel   | Reflate again
	choke1 = (ed > 4)             ? choke1.mt_expand( mode="vertical", U=RCrg,V=RCrg ) : choke1   #      . . . . . x x x    1 pixel  /

	# Over-dilation - extra reflation up to about 1 pixel
	choke1 = (od == 0)            ? choke1 : \
	         (od == 1)            ? choke1.QTGMC_inflate( U=RCrg,V=RCrg ) : \
	         (od == 2)            ? choke1.QTGMC_inflate( U=RCrg,V=RCrg ).QTGMC_inflate( U=RCrg,V=RCrg ) : \
	                                choke1.mt_expand ( U=RCrg,V=RCrg )

	# Areas of negative difference (similar to above)
	choke2 =                        diff.  mt_expand( mode="vertical", U=RCrg,V=RCrg )         
	choke2 = (ed > 2)             ? choke2.mt_expand( mode="vertical", U=RCrg,V=RCrg ) : choke2
	choke2 = (ed > 5)             ? choke2.mt_expand( mode="vertical", U=RCrg,V=RCrg ) : choke2
	choke2 = (ed % 3 != 0)        ? choke2.QTGMC_inflate( U=RCrg,V=RCrg )              : choke2
	choke2 = (ed == 2 || ed == 5) ? choke2.RemoveGrain( 4              )               : choke2
	choke2 =                        choke2.mt_inpand( mode="vertical", U=RCrg,V=RCrg )
	choke2 = (ed > 1)             ? choke2.mt_inpand( mode="vertical", U=RCrg,V=RCrg ) : choke2
	choke2 = (ed > 4)             ? choke2.mt_inpand( mode="vertical", U=RCrg,V=RCrg ) : choke2
	choke2 = (od == 0)            ? choke2 : \
	         (od == 1)            ? choke2.QTGMC_deflate( U=RCrg,V=RCrg )  : \
	         (od == 2)            ? choke2.QTGMC_deflate( U=RCrg,V=RCrg ).QTGMC_deflate( U=RCrg,V=RCrg ) : \
	                                choke2.mt_inpand ( U=RCrg,V=RCrg )

	# Combine above areas to find those areas of difference to restore
	restore = diff.mt_lutxy( choke1, "x 129 scalef < x y range_half < range_half y ? ?", U=RCrg,V=RCrg ).mt_lutxy( choke2, "x 127 scalef > x y range_half > range_half y ? ?", U=RCrg,V=RCrg )

	return Input.mt_adddiff( restore, U=RCrgo,V=RCrgo )
}

