// by Yasuhiro Fujii <y-fujii@mimosa-pudica.net>, public domain
#pragma once

#include <iterator>
#include <limits>
#include <cstddef>
#include <cassert>
#include <emmintrin.h>
//#include <pmmintrin.h>
#if defined( __SSE4_1__ )
#include <smmintrin.h>
#endif

namespace ssp {

struct one  {};
struct zero {};
static one  const I;
static zero const O;

template<class, int> struct reference;
template<> struct reference<int32_t, 4>;
template<> struct reference<float, 4>;

template<class, int> struct array;
template<> struct array<int32_t, 4>;
template<> struct array<float, 4>;

template<class Elem, int N>
struct reference {
	template<class Rhs>
	Rhs const& operator=( Rhs const& rhs ) {
		for( int i = 0; i < N; ++i ) {
			*_data[i] = rhs._data[i];
		}
		return rhs;
	}

	operator array<Elem, N>() const {
		array<Elem, N> dst;
		for( int i = 0; i < N; ++i ) {
			dst._data[i] = *_data[i];
		}
		return dst;
	}

	template<class T, class U>
	reference<U, N> member( U T::*m ) const {
		reference<U, N> ref;
		for( int i = 0; i < N; ++i ) {
			ref._data[i] = &(_data[i]->*m);
		}
		return ref;
	}

	//private:
		Elem* _data[N];
};

template<class Elem, int N>
struct reference<Elem const, N> {
	operator array<Elem, N>() const {
		array<Elem, N> dst;
		for( int i = 0; i < N; ++i ) {
			dst._data[i] = *_data[i];
		}
		return dst;
	}

	template<class T, class U>
	reference<U const, N> member( U T::*m ) const {
		reference<U const, N> ref;
		for( int i = 0; i < N; ++i ) {
			ref._data[i] = &(_data[i]->*m);
		}
		return ref;
	}

	//private:
		Elem const* _data[N];
};

template<class Elem, int N>
struct array {
	array() {
	}

	array( Elem x ) {
		for( int i = 0; i < N; ++i ) {
			_data[i] = x;
		}
	}

	template<class T, class U>
	reference<U, N> member( U T::*m ) const {
		reference<U, N> ref;
		for( int i = 0; i < N; ++i ) {
			ref._data[i] = &(_data[i].*m);
		}
		return ref;
	}

	//private:
		Elem _data[N];
};

template<>
struct array<int32_t, 4> {
	array() {}

	array( int32_t x, int32_t y, int32_t z, int32_t w ) {
		_packed = _mm_set_epi32( w, z, y, x );
	}

	array( int32_t x ) {
		_packed = _mm_set1_epi32( x );
	}

	array( one ) {
		_packed = _mm_set1_epi32( 1 );
	}

	array( zero ) {
		_packed = _mm_setzero_si128();
	}

	explicit array( __m128i xs ):
		_packed( xs ) {}

	explicit array( array<float, 4> const& );

	//private:
		union {
			__m128i _packed;
			int32_t _data[4];
		};
};

template<>
struct array<float, 4> {
	array() {}

	array( float x, float y, float z, float w ) {
		_packed = _mm_set_ps( w, z, y, x );
	}

	array( float x ) {
		_packed = _mm_set1_ps( x );
	}

	array( one ) {
		_packed = _mm_set1_ps( 1.0f );
	}

	array( zero ) {
		_packed = _mm_setzero_ps();
	}

	explicit array( __m128 xs ):
		_packed( xs ) {}

	explicit array( array<int32_t, 4> const& );

	//private:
		union {
			__m128 _packed;
			float _data[4];
		};
};

inline array<int32_t, 4>::array( array<float, 4> const& xs ) {
	_packed = _mm_cvtps_epi32( xs._packed );
}

inline array<float, 4>::array( array<int32_t, 4> const& xs ) {
	_packed = _mm_cvtepi32_ps( xs._packed );
}

template<>
struct reference<float, 4> {
	array<float, 4> const& operator=( array<float, 4> const& rhs ) {
		// Why is this fastest?
		union {
			float array[4];
			__m128 packed;
		} y;
		y.packed = rhs._packed;
		*_data[0] = y.array[0];
		*_data[1] = y.array[1];
		*_data[2] = y.array[2];
		*_data[3] = y.array[3];
		/*
		*_data[0] = rhs._data[0];
		*_data[1] = rhs._data[1];
		*_data[2] = rhs._data[2];
		*_data[3] = rhs._data[3];
		*/
		return rhs;
	}

	operator array<float, 4>() const {
		return array<float, 4>(
			*_data[0],
			*_data[1],
			*_data[2],
			*_data[3]
		);
	}

	//private:
		float* _data[4];
};

template<>
struct reference<int32_t, 4> {
	array<int32_t, 4> const& operator=( array<int32_t, 4> const& rhs ) {
		union {
			int32_t array[4];
			__m128i packed;
		} y;
		y.packed = rhs._packed;
		*_data[0] = y.array[0];
		*_data[1] = y.array[1];
		*_data[2] = y.array[2];
		*_data[3] = y.array[3];
		/*
		*_data[0] = rhs._data[0];
		*_data[1] = rhs._data[1];
		*_data[2] = rhs._data[2];
		*_data[3] = rhs._data[3];
		*/
		return rhs;
	}

	operator array<int32_t, 4>() const {
		return array<int32_t, 4>(
			*_data[0],
			*_data[1],
			*_data[2],
			*_data[3]
		);
	}

	//private:
		int32_t* _data[4];
};

template<class T, class U, int N> array<T, N> bitcast( array<U, N> const& );

template<>
array<float, 4> bitcast<float>( array<int32_t, 4> const& x ) {
	return array<float, 4>( _mm_castsi128_ps( x._packed ) );
}

template<>
array<int32_t, 4> bitcast<int32_t>( array<float, 4> const& x ) {
	return array<int32_t, 4>( _mm_castps_si128( x._packed ) );
}

/*
template<class T, int N>
inline array<T, N> where(
	array<int32_t, N> const& c0, array<T, N> const& x0,
	array<int32_t, N> const& c1, array<T, N> const& x1,
	array<T, N> const& y
) {
	return where( c0, x0, where( c1, x1, y ) );
}
*/

template<class T, int N, class... Args>
inline array<T, N> where(
	array<int32_t, N> const& c0, array<T, N> const& x0,
	array<int32_t, N> const& c1, array<T, N> const& x1,
	Args... args
) {
	return where( c0, x0, where( c1, x1, args... ) );
}

inline array<int32_t, 4> where( array<int32_t, 4> const& c, array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
#if defined( __SSE4_1__ )
	__m128i z = _mm_blendv_epi8( y._packed, x._packed, c._packed );
#else
	// return (c & x) | (~c & y);
	__m128i z = _mm_or_si128(
		_mm_and_si128   ( c._packed, x._packed ),
		_mm_andnot_si128( c._packed, y._packed )
	);
#endif
	return array<int32_t, 4>( z );
}

inline array<float, 4> where( array<int32_t, 4> const& c, array<float, 4> const& x, array<float, 4> const& y ) {
#if defined( __SSE4_1__ )
	__m128 z = _mm_blendv_ps( y._packed, x._packed, _mm_castsi128_ps( c._packed ) );
#else
	__m128 z = _mm_or_ps(
		_mm_and_ps   ( _mm_castsi128_ps( c._packed ), x._packed ),
		_mm_andnot_ps( _mm_castsi128_ps( c._packed ), y._packed )
	);
#endif
	return array<float, 4>( z );
}

inline array<int32_t, 4> operator+( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_add_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator-( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_sub_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator+( array<int32_t, 4> const& x ) {
	return x;
}

inline array<int32_t, 4> operator-( array<int32_t, 4> const& x ) {
	__m128i z = _mm_sub_epi32( _mm_setzero_si128(), x._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator*( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
#if defined( __SSE4_1__ )
	__m128i z = _mm_mullo_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
#elif 1
	__m128i ylo = y._packed;
	ylo = _mm_shufflelo_epi16( ylo, _MM_SHUFFLE( 0, 0, 2, 2 ) );
	ylo = _mm_shufflehi_epi16( ylo, _MM_SHUFFLE( 0, 0, 2, 2 ) );
	__m128i yhi = y._packed;
	yhi = _mm_shufflelo_epi16( yhi, _MM_SHUFFLE( 1, 1, 3, 3 ) );
	yhi = _mm_shufflehi_epi16( yhi, _MM_SHUFFLE( 1, 1, 3, 3 ) );
	__m128i z = _mm_add_epi16(
		_mm_mullo_epi16( x._packed, ylo ),
		_mm_slli_epi32(
			_mm_add_epi16(
				_mm_mulhi_epu16( x._packed, ylo ),
				_mm_mullo_epi16( x._packed, yhi )
			),
			16
		)
	);
	return array<int32_t, 4>( z );
#elif 1
	__m128i ff = _mm_cmpeq_epi32( ff, ff );
	__m128i mask = _mm_srli_epi64( ff, 32 );

	__m128i zlo = _mm_mul_epu32( x._packed, y._packed );
	__m128i zhi = _mm_mul_epu32(
		_mm_srli_si128( x._packed, 4 ),
		_mm_srli_si128( y._packed, 4 )
	);
	zlo = _mm_and_si128( zlo, mask );
	zhi = _mm_slli_si128( _mm_and_si128( zhi, mask ), 4 );

	__m128i z = _mm_or_si128( zlo, zhi );
	return array<int32_t, 4>( z );
#else
	return array<int32_t, 4>(
		x._data[0] * y._data[0],
		x._data[1] * y._data[1],
		x._data[2] * y._data[2],
		x._data[3] * y._data[3]
	);
#endif
}

inline array<int32_t, 4> operator/( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	return array<int32_t, 4>(
		x._data[0] / y._data[0],
		x._data[1] / y._data[1],
		x._data[2] / y._data[2],
		x._data[3] / y._data[3]
	);
}

inline array<int32_t, 4> operator%( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	return array<int32_t, 4>(
		x._data[0] % y._data[0],
		x._data[1] % y._data[1],
		x._data[2] % y._data[2],
		x._data[3] % y._data[3]
	);
}

inline array<int32_t, 4> operator&( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_and_si128( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator|( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_or_si128( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator^( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_xor_si128( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator==( array<int32_t, 4> const&, array<int32_t, 4> const& );

inline array<int32_t, 4> operator~( array<int32_t, 4> const& x ) {
	return x ^ (x == x);
}

inline array<int32_t, 4> operator<<( array<int32_t, 4> const& x, int32_t y ) {
	__m128i z = _mm_slli_epi32( x._packed, y );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator>>( array<int32_t, 4> const& x, int32_t y ) {
	__m128i z = _mm_srai_epi32( x._packed, y );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> lsr( array<int32_t, 4> const& x, int32_t y ) {
	__m128i z = _mm_srli_epi32( x._packed, y );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator>>( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_sra_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator<<( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_sll_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> lsr( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_srl_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator==( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_cmpeq_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator!=( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	// return _mm_cmpneq_epi32( x._packed, y._packed );
	return ~(x == y);
}

inline array<int32_t, 4> operator<( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_cmplt_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator>( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	__m128i z = _mm_cmpgt_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator<=( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	//return _mm_cmple_epi32( x._packed, y._packed );
	return ~(x > y);
}

inline array<int32_t, 4> operator>=( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	//return _mm_cmpge_epi32( x._packed, y._packed );
	return ~(x < y);
}

inline array<int32_t, 4> min( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
#if defined( __SSE4_1__ )
	__m128i z = _mm_min_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
#else
	return where( x < y, x, y );
#endif
}

inline array<int32_t, 4> max( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
#if defined( __SSE4_1__ )
	__m128i z = _mm_max_epi32( x._packed, y._packed );
	return array<int32_t, 4>( z );
#else
	return where( x < y, y, x );
#endif
}

inline array<int32_t, 4> abs( array<int32_t, 4> const& x ) {
#if defined( __SSSE3__ )
	__m128i z = _mm_abs_epi32( x._packed );
	return array<int32_t, 4>( z );
#else
	return where( x < O, -x, +x );
#endif
}

inline bool any_of( array<int32_t, 4> const& x ) {
	// return _mm_movemask_epi8( (x != 0)._packed ) != 0x0000;
	return _mm_movemask_epi8( x._packed ) != 0x0000;
}

inline bool all_of( array<int32_t, 4> const& x ) {
	// return _mm_movemask_epi8( (x != 0)._packed ) == 0xffff;
	return _mm_movemask_epi8( x._packed ) == 0xffff;
}

inline bool none_of( array<int32_t, 4> const& x ) {
	// return _mm_movemask_epi8( (x != 0)._packed ) == 0x0000;
	return _mm_movemask_epi8( x._packed ) == 0x0000;
}

inline bool nall_of( array<int32_t, 4> const& x ) {
	// return _mm_movemask_epi8( (x != 0)._packed ) != 0xffff;
	return _mm_movemask_epi8( x._packed ) != 0xffff;
}

inline array<float, 4> operator+( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_add_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<float, 4> operator-( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_sub_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<float, 4> operator+( array<float, 4> const& x ) {
	return x;
}

inline array<float, 4> operator-( array<float, 4> const& x ) {
	__m128 z = _mm_sub_ps( _mm_setzero_ps(), x._packed );
	return array<float, 4>( z );
}

inline array<float, 4> operator*( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_mul_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<float, 4> operator/( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_div_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<int32_t, 4> operator==( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmpeq_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator!=( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmpneq_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator<( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmplt_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator>( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmpgt_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator<=( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmple_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<int32_t, 4> operator>=( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z = _mm_castps_si128( _mm_cmpge_ps( x._packed, y._packed ) );
	return array<int32_t, 4>( z );
}

inline array<float, 4> sqrt( array<float, 4> const& x ) {
	__m128 z = _mm_sqrt_ps( x._packed );
	return array<float, 4>( z );
}

inline array<float, 4> abs( array<float, 4> const& x ) {
	int32_t mask = ~(1 << 31);
	return bitcast<float>( bitcast<int32_t>( x ) & mask );
}

inline array<int32_t, 4> signbit( array<float, 4> const& x ) {
	return (bitcast<int32_t>( x ) >> 31) & 0x1;
}

inline array<float, 4> frexp( array<float, 4> const& x, array<int32_t, 4>* _e ) {
	array<int32_t, 4> xi = bitcast<int32_t>( x );
	array<int32_t, 4> e = (xi >> 23) & 0xff;
	array<int32_t, 4> f = xi & 0x807fffff;

	array<int32_t, 4> isDenorm = (e == 0);
	if( any_of( isDenorm ) ) {
		array<int32_t, 4> xb = bitcast<int32_t>( x * 16777216.0f );
		e = where( isDenorm, ((xb >> 23) & 0xff) - 24, e );
		f = where( isDenorm, xb & 0x807fffff, f );
	}

	*_e = e - 0x7e;
	return bitcast<float>( f | (0x7e << 23) );
}

inline array<float, 4> ldexp( array<float, 4> const& x, array<int32_t, 4> const& n ) {
	// XXX
	return x * bitcast<float>( (n + 0x7f) << 23 );
}

inline array<float, 4> copysign( array<float, 4> const& x, array<float, 4> const& y ) {
	int32_t mask_y = 1 << 31;
	int32_t mask_x = ~mask_y;
	return bitcast<float>(
		(bitcast<int32_t>( x ) & mask_x) | (bitcast<int32_t>( y ) & mask_y)
	);
}

inline array<float, 4> truncate( array<float, 4> const& x ) {
	array<int32_t, 4> e;
	frexp( x, &e );
	return where(
		e <=  0, array<float, 4>( +0.0f ),
		e <= 24, bitcast<float>( bitcast<int32_t>( x ) & ((e == e) << (24 - e)) ),
		        x // include NaN
	);
}

inline array<int32_t, 4> truncatei( array<float, 4> const& x ) {
	return array<int32_t, 4>( _mm_cvttps_epi32( x._packed ) );
}

inline array<float, 4> floor( array<float, 4> const& x ) {
#if defined( __SSE4_1__ )
	__m128 z = _mm_floor_ps( x._packed );
	return array<float, 4>( z );
#else
	array<float, 4> z = truncate( x );
	return where( z > x, z - I, z );
#endif
}

inline array<int32_t, 4> floori( array<float, 4> const& x ) {
	array<int32_t, 4> z = truncatei( x );
	return where( array<float, 4>( z ) > x, z - I, z );
}

inline array<float, 4> ceil( array<float, 4> const& x ) {
#if defined( __SSE4_1__ )
	__m128 z = _mm_ceil_ps( x._packed );
	return array<float, 4>( z );
#else
	array<float, 4> z = truncate( x );
	return where( z < x, z + I, z );
#endif
}

inline array<int32_t, 4> ceili( array<float, 4> const& x ) {
	array<int32_t, 4> z = truncatei( x );
	return where( array<float, 4>( z ) < x, z + I, z );
}

inline array<float, 4> min( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_min_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<float, 4> max( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_max_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

template<class Result, class Func, int N, class... Ts>
array<Result, N> call( Func const& f, array<Ts, N> const&... args ) {
	array<Result, N> result;
	for( int i = 0; i < N; ++i ) {
		result._data[i] = f( args._data[i]... );
	}
	return result;
}

template<class Container, class Elem>
struct container_view {
	container_view( Container& c ):
		_container( c ) {}

	template<class T, int N>
	reference<Elem, N> operator[]( array<T, N> const& idx ) const {
		reference<Elem, N> ref;
		for( int i = 0; i < N; ++i ) {
			ref._data[i] = &_container[idx._data[i]];
		}
		return ref;
	}

	/*
	template<class T>
	inline reference<Elem, 4> operator[]( array<T, 4> const& idx ) const {
		reference<Elem, 4> ref;
		ref._data[0] = &(*_container)[idx._data[0]];
		ref._data[1] = &(*_container)[idx._data[1]];
		ref._data[2] = &(*_container)[idx._data[2]];
		ref._data[3] = &(*_container)[idx._data[3]];
		return ref;
	}
	*/

	private:
		Container& _container;
};

template<class T>
container_view<T, typename T::value_type> view( T& c ) {
	return container_view<T, typename T::value_type>( c );
}

template<class T>
container_view<T const, typename T::value_type const> const_view( T const& c ) {
	return container_view<T const, typename T::value_type const>( c );
}


template<int simd_size, int parallel_type> struct Runner;

template<>
struct Runner<4, 0> {
	enum schedule_method {
		schedule_static,
		schedule_dynamic,
		schedule_guided,
	};

	static int const vector_size = 4;
	typedef array<int32_t, 4> a_int32;
	typedef array<float,   4> a_float;

	template<class Func>
	void for_1d( int32_t x0, int32_t x1, Func const& f ) {
		a_int32 xs( 0, 1, 2, 3 );

		#pragma omp parallel for schedule(guided, 1)
		for( int32_t x = x0; x < x1; x += 4 ) {
			f( xs + x );
		}
	}

	template<class Func>
	void for_2d( int32_t x0, int32_t x1, int32_t y0, int32_t y1, Func const& f ) {
		a_int32 ys( 0, 0, 1, 1 );
		a_int32 xs( 0, 1, 0, 1 );
		#pragma omp parallel for schedule(guided, 1)
		for( int32_t y = y0; y < y1; y += 2 ) {
			a_int32 yi = ys + y;
			for( int32_t x = x0; x < x1; x += 2 ) {
				f( xs + x, yi );
			}
		}
	}
};


}
