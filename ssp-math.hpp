#pragma once

#include <limits>
#include <cmath>
#include "ssp.hpp"

namespace ssp {


namespace detail {
	#define USE_SSE2
	#include "sse_mathfun.h"
	#undef USE_SSE2
}

inline array<float, 4> sin( array<float, 4> const& x ) {
	return array<float, 4>( detail::sin_ps( x._packed ) );
}

inline array<float, 4> cos( array<float, 4> const& x ) {
	return array<float, 4>( detail::cos_ps( x._packed ) );
}

inline array<float, 4> exp( array<float, 4> const& x ) {
	return array<float, 4>( detail::exp_ps( x._packed ) );
}

inline array<float, 4> log( array<float, 4> const& x ) {
	return array<float, 4>( detail::log_ps( x._packed ) );
}

template<int N>
inline array<float, N> asin_core( array<float, N> const& x1, array<float, N> const& x2 ) {
	array<float, N> z =
		((((4.2163199048e-2f  * x2
		  + 2.4181311049e-2f) * x2
		  + 4.5470025998e-2f) * x2
		  + 7.4953002686e-2f) * x2
		  + 1.6666752422e-1f) * x2 * x1
		  + x1;

	return where( x2 < 1e-8f, x1, z );
}

template<int N>
inline array<float, N> asin( array<float, N> const& x ) {
	array<float, N> a = abs( x );
	array<int32_t, N> flag = a > 0.5f;
	array<float, N> x2 = where( flag, 0.5f * (1.0f - a), a * a );
	array<float, N> x1 = where( flag, sqrt( x2 ), a );

	array<float, N> z = asin_core( x1, x2 );

	z = where( flag, float( M_PI / 2.0 ) - (z + z), z );
	z = copysign( z, x );
	return z;
}

template<int N>
inline array<float, N> acos( array<float, N> const& x ) {
	array<float, N> a = abs( x );
	array<int32_t, N> flag = a > 0.5f;
	array<float, N> x2 = where( flag, 0.5f * (1.0f - a), a * a );
	array<float, N> x1 = where( flag, sqrt( x2 ), x );

	array<float, N> z = asin_core( x1, x2 );

	return where(
		x < -0.5f, float( M_PI ) - (z + z),
		x > +0.5f,                 (z + z),
		           float( M_PI / 2.0 ) - z // include NaN
	);
}

template<int N>
inline array<float, N> atan( array<float, N> const& x ) {
	array<float, N> a = abs( x );
	array<int32_t, N> flag0 = a > float( 3.0 * M_PI / 8.0 );
	array<int32_t, N> flag1 = a > float( 1.0 * M_PI / 8.0 );

	a = where(
		flag0, -1.0f / a,
		flag1, (a - 1.0f) / (a + 1.0f),
		       a
	);

	array<float, N> a2 = a * a;
	array<float, N> z =
		(((8.05374449538e-2  * a2
		 - 1.38776856032e-1) * a2
		 + 1.99777106478e-1) * a2
		 - 3.33329491539e-1) * a2 * a
		 + a;

	z = z + where(
		flag0, array<float, N>( M_PI / 2.0 ),
		flag1, array<float, N>( M_PI / 4.0 ),
		       array<float, N>( 0.0f )
	);
	z = copysign( z, x );

	return z;
}

template<int N>
array<float, N> atan2( array<float, N> const& y, array<float, N> const& x ) {
	array<float, N> z = atan( y / x );

	return where( x > 0.0f, copysign( M_PI, y ) + z, z );
}

template<int N>
array<float, N> sinh( array<float, N> const& x ) {
	array<float, N> x2 = x * x;
	array<float, N> z0 =
		((2.03721912945e-4  * x2
		+ 8.33028376239e-3) * x2
		+ 1.66667160211e-1) * x2 * x
		+ x;

	array<float, N> a = abs( x );
	array<float, N> z1 = exp( a );
	z1 = 0.5f * z1 - 0.5f / z1;
	z1 = copysign( z1, x );

	return where( a <= 1.0f, z0, z1 );
}

template<int N>
array<float, N> cosh( array<float, N> const& x ) {
	array<float, N> z = exp( abs( x ) );
	return 0.5f * z + 0.5f / z;
}

template<int N>
array<float, N> tanh( array<float, N> const& x ) {
	array<float, N> x2 = x * x;
	array<float, N> z0 =
		((((- 5.70498872745e-3  * x2
			+ 2.06390887954e-2) * x2
			- 5.37397155531e-2) * x2
			+ 1.33314422036e-1) * x2
			- 3.33332819422e-1) * x2 * x
			+ x;

	array<float, N> a = abs( x );
	array<float, N> z1 = 1.0f - 2.0f / (exp( a + a ) + 1.0f);
	z1 = copysign( z1, x );

	return where( a < 0.625f, z0, z1 );
}


}
