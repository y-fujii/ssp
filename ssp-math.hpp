#pragma once

#include <limits>
#include <cmath>
#include "ssp.hpp"

namespace ssp {


template<int N>
inline array<float, N> exp( array<float, N> const& x ) {
	static float const ilog2 = 1.0 / std::log( 2.0 );
	static float const log2_0 = std::ldexp( std::floor( std::ldexp( std::log( 2.0 ), 16 ) ), -16 );
	static float const log2_1 = std::log( 2.0 ) - log2_0;

	array<float, N> n = floor( ilog2 * x + 0.5f );
	array<float, N> r = x - n * log2_0 - n * log2_1;

	array<float, N> z =
		(((((1.9875691500e-4  * r
		   + 1.3981999507e-3) * r
		   + 8.3334519073e-3) * r
		   + 4.1665795894e-2) * r
		   + 1.6666665459e-1) * r
		   + 5.0000001201e-1) * (r * r)
		   + r
		   + 1.0f;

	return ldexp( z, array<int32_t, N>( n ) );
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

	return z;
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
	static float const tan_3pi_8 = std::tan( 3.0 * M_PI / 8.0 );
	static float const tan_1pi_8 = std::tan( 1.0 * M_PI / 8.0 );

	array<float, N> a = abs( x );
	array<int32_t, N> flag0 = a > tan_3pi_8;
	array<int32_t, N> flag1 = a > tan_1pi_8;

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

template<int N>
void rem_pio4( array<float, N> const& x, array<int32_t, N>& n, array<float, N>& r ) {
	n = floori( float( 4.0 / M_PI ) * x );
	n = n + (n & 1);
	array<float, N> nf( n );
	r = x - nf * 0.78515625
	      - nf * 2.4187564849853515625e-4
	      - nf * 3.77489497744594108e-8;
	/*
	r = x - nf * 0.7853851318359375
	      - nf * 1.30315311253070831298828125e-5
	      - nf * 3.03855025325309630e-11;
	*/
}

template<int N>
array<float, N> sin_core( array<float, N> const& x1, array<float, N> const& x2 ) {
	array<float, N> z =
		((- 1.9515295891e-4  * x2
		  + 8.3321608736e-3) * x2
		  - 1.6666654611e-1) * x2 * x1
		  + x1;

	// z = where( x2 < 1e-8, x1, z );
	return z;
}

template<int N>
array<float, N> cos_core( array<float, N> const&, array<float, N> const& x2 ) {
	array<float, N> z =
		((2.443315711809948e-5  * x2
		- 1.388731625493765e-3) * x2
		+ 4.166664568298827e-2) * (x2 * x2)
		- 0.5f * x2
		+ 1.0f;

	return z;
}

template<int N>
array<float, N> sin( array<float, N> const& x ) {
	array<float, N> r;
	array<int32_t, N> n;
	rem_pio4( x, n, r );

	array<float, N> r2 = r * r;
	array<float, N> z = where( (n & 3) != 0,
		cos_core( r, r2 ),
		sin_core( r, r2 )
	);

	z = where( (n & 4) != 0, -z, +z );
	z = where( abs( x ) < 8192.0f, z, 0.0f );
	return z;
}

template<int N>
array<float, N> cos( array<float, N> const& x ) {
	array<float, N> r;
	array<int32_t, N> n;
	rem_pio4( x, n, r );

	array<float, N> r2 = r * r;
	array<float, N> z = where( (n & 3) != 0,
		sin_core( r, r2 ),
		cos_core( r, r2 )
	);

	z = where( ((n + 2) & 4) != 0, -z, +z );
	z = where( abs( x ) < 8192.0f, z, 0.0f );
	return z;
}

template<int N>
array<float, N> tan( array<float, N> const& x ) {
	array<float, N> r;
	array<int32_t, N> n;
	rem_pio4( x, n, r );

	array<float, N> r2 = r * r;
	array<float, N> z =
		(((((9.38540185543e-3  * r2
		   + 3.11992232697e-3) * r2
		   + 2.44301354525e-2) * r2
		   + 5.34112807005e-2) * r2
		   + 1.33387994085e-1) * r2
		   + 3.33331568548e-1) * r2 * r
		   + r;

	z = where( (n & 2) != 0, -1.0f / z, z );
	z = where( abs( x ) < 8192.0f, z, 0.0f );
	return z;
}

template<int N>
array<float, N> log( array<float, N> const& x ) {
	static float const sqrt2h = std::sqrt( 2.0 ) / 2.0;
	static float const log2_0 = std::ldexp( std::floor( std::ldexp( std::log( 2.0 ), 16 ) ), -16 );
	static float const log2_1 = std::log( 2.0 ) - log2_0;

	array<int32_t, N> n;
	array<float, N> f = frexp( x, &n );
	n = where( f < sqrt2h, n - 1, n );
	f = where( f < sqrt2h, f + f, f ) - 1.0f;

	array<float, N> f2 = f * f;
	array<float, N> z =
		((((((((7.0376836292e-2  * f
	          - 1.1514610310e-1) * f
	          + 1.1676998740e-1) * f
	          - 1.2420140846e-1) * f
	          + 1.4249322787e-1) * f
	          - 1.6668057665e-1) * f
	          + 2.0000714765e-1) * f
	          - 2.4999993993e-1) * f
	          + 3.3333331174e-1) * f * f2;

	array<float, N> e( n );
	z = z + log2_1 * e;
	z = z - 0.5f * f2 + f;
	z = z + log2_0 * e;

	z = where(
		x <  0.0f, array<float, N>( +0.0f / +0.0f ),
		x == 0.0f, array<float, N>( -1.0f / +0.0f ),
		z
	);
	return z;
}


}
