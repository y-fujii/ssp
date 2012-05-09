#pragma once

#include <limits>
#include <cmath>
#include "ssp.hpp"

namespace detail {
	#define USE_SSE2
	#include "sse_mathfun.h"
	#undef USE_SSE2
}

inline ssp::array<float, 4> sin( ssp::array<float, 4> const& x ) {
	return ssp::array<float, 4>( detail::sin_ps( x._packed ) );
}

inline ssp::array<float, 4> cos( ssp::array<float, 4> const& x ) {
	return ssp::array<float, 4>( detail::cos_ps( x._packed ) );
}

inline ssp::array<float, 4> exp( ssp::array<float, 4> const& x ) {
	return ssp::array<float, 4>( detail::exp_ps( x._packed ) );
}

inline ssp::array<float, 4> log( ssp::array<float, 4> const& x ) {
	return ssp::array<float, 4>( detail::log_ps( x._packed ) );
}

template<int N>
inline ssp::array<float, N> asin( ssp::array<float, N> const& x ) {
	using namespace ssp;

	array<float, N> a = abs( x );
	array<float, N> s = where( x > 0.0f, array<float, N>( +1.0f ), -1.0f );

	array<int32_t, N> flag = a > 0.5f;

	array<float, N> x2 = where( flag, 0.5f * (1.0f - a), a * a );
	array<float, N> x1 = where( flag, sqrt( x2 ), a );

	array<float, N> z =
		((((4.2163199048e-2f  * x2
	      + 2.4181311049e-2f) * x2
	      + 4.5470025998e-2f) * x2
	      + 7.4953002686e-2f) * x2
		  + 1.6666752422e-1f) * x2 * x1
	      + x1;

	z = where( flag, float(M_PI / 2.0) - (z + z), z );
	z = where( a < 1e-4f, a, z );
	z = s * z;
	z = where( a > 1.0f, std::numeric_limits<float>::quiet_NaN(), z );
	return z;
}


template<int N>
inline ssp::array<float, N> acos( ssp::array<float, N> const& x ) {
	using namespace ssp;

	array<int32_t, N> flag0 = -1.0f <= x & x <  -0.5f;
	array<int32_t, N> flag1 = +0.5f <  x & x <= +1.0f;
	array<int32_t, N> flag2 = -0.5f <= x & x <  +0.5f;
	// array<int32_t, N> flag3 = x < -1.0f | +1.0f < x;

	array<float, N> z = sqrt( 0.5f * (1.0f - abs( x )) );
	array<float, N> u = where( flag0 | flag1, z, x );

	array<float, N> v = asin( u );

	array<float, N> w = where(
		flag0, float( M_PI ) - 2.0f * v,
		flag1,                 2.0f * v,
		flag2, float( M_PI / 2.0 ) - v,
		std::numeric_limits<float>::quiet_NaN()
	);
	return w;
}

/*
float atan( float x ) {
	float sign = where( x < 0.0f, -1.0f, +1.0f );
	float x    = where( x < 0.0f, -x   , +x    );

	y = where(
		x > (3.0 * M_PI / 8.0), (M_PI / 2.0),
		x > (1.0 * M_PI / 8.0), (M_PI / 4.0),
								0.0f
	);

	x = where(
		x > (3.0 * M_PI / 8.0), -1.0f / x,
		x > (1.0 * M_PI / 8.0), (x - 1.0f) / (x + 1.0f),
		                        x
	);

	float z = x * x;
	y += (((8.05374449538e-2  * z
	      - 1.38776856032E-1) * z
	      + 1.99777106478E-1) * z
	      - 3.33329491539E-1) * z * x
	      + x;

	y = s * y;
	return y;
}

float atan2( float y, float x ) {
	float z = atanf( y / x );

	z = where( y < 0.0f,
			where( x < 0.0f, -M_PI + z, z ),
			where( x < 0.0f, +M_PI + z, z )
		);

	z.assign_if( x == 0.0f,
		where( y <  0.0f, -M_PI / 2.0,
		where( y == 0.0f, 0.0f,
		where( y >  0.0f, +M_PI / 2.0
	))));

	z.assign_if( y == 0.0f,
		where( x < 0.0f, M_PI, 0.0f )
	);

	return z;
}
*/
