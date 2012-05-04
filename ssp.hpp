#pragma once

#include <iterator>
#include <cstddef>
#include <cassert>
#include <emmintrin.h>
//#include <pmmintrin.h>
#if defined( __SSE4_1__ )
#include <smmintrin.h>
#endif

namespace ssp {


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
struct array {
	array() {
	}

	array( Elem x ) {
		for( int i = 0; i < N; ++i ) {
			_data[i] = x;
		}
	}

	template<class T, class U>
	reference<U, N> member( U T::*m ) {
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

template<class T, int N>
array<T, N> where( array<int32_t, N> const&, array<T, N> const&, array<T, N> const& );

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

#if defined( __SSE4_1__ )
inline array<float, 4> floor( array<float, 4> const& x ) {
	__m128 z = _mm_floor_ps( x._packed );
	return array<float, 4>( z );
}

inline array<float, 4> ceil( array<float, 4> const& x ) {
	__m128 z = _mm_ceil_ps( x._packed );
	return array<float, 4>( z );
}
#endif

inline array<float, 4> min( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_min_ps( x._packed, y._packed );
	return array<float, 4>( z );
}

inline array<float, 4> max( array<float, 4> const& x, array<float, 4> const& y ) {
	__m128 z = _mm_max_ps( x._packed, y._packed );
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
#else
	return array<int32_t, 4>(
		x._data[0] * y._data[0],
		x._data[1] * y._data[1],
		x._data[2] * y._data[2],
		x._data[3] * y._data[3]
	);
#endif
}

/*
inline array<int32_t, 4> operator/( array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	//return _mm_div_epi32( x._packed, y._packed );
	return array<int32_t, 4>(
		x._data[0] / y._data[0],
		x._data[1] / y._data[1],
		x._data[2] / y._data[2],
		x._data[3] / y._data[3]
	);
}
*/

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

inline array<int32_t, 4> operator~( array<int32_t, 4> const& x ) {
	#pragma GCC diagnostic ignored "-Wuninitialized"
	__m128i t;
	__m128i z = _mm_xor_si128(
		x._packed,
		_mm_cmpeq_epi32( t, t )
	);
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

inline bool any_of( array<int32_t, 4> const& x ) {
	/*
	union {
		int64_t array[2];
		__m128i packed;
	} y;
	_mm_store_si128( &y.packed, x._packed );
	return (y.array[0] | y.array[1]) != 0l;
	*/
	return _mm_movemask_epi8( x._packed ) != 0x0000;
}

inline bool all_of( array<int32_t, 4> const& x ) {
	/*
	union {
		int64_t array[2];
		__m128i packed;
	} y;
	_mm_store_si128( &y.packed, x._packed );
	return ~(y.array[0] & y.array[1]) == 0l;
	*/
	return _mm_movemask_epi8( x._packed ) == 0xffff;
}

inline bool none_of( array<int32_t, 4> const& x ) {
	return _mm_movemask_epi8( x._packed ) == 0x0000;
}

inline bool nall_of( array<int32_t, 4> const& x ) {
	return _mm_movemask_epi8( x._packed ) != 0xffff;
}

template<class T, int N>
array<T, N> where( array<int32_t, N> const&, array<T, N> const&, array<T, N> const& );

template<>
inline array<int32_t, 4> where( array<int32_t, 4> const& c, array<int32_t, 4> const& x, array<int32_t, 4> const& y ) {
	// return (c & x) | (~c & y);
	__m128i z0 = _mm_and_si128( c._packed, x._packed );
	__m128i z1 = _mm_andnot_si128( c._packed, y._packed );
	__m128i z2 = _mm_or_si128( z0, z1 );
	return array<int32_t, 4>( z2 );
}

template<>
inline array<float, 4> where( array<int32_t, 4> const& c, array<float, 4> const& x, array<float, 4> const& y ) {
	__m128i z0 = _mm_and_si128( c._packed, _mm_castps_si128( x._packed ) );
	__m128i z1 = _mm_andnot_si128( c._packed, _mm_castps_si128( y._packed ) );
	__m128  z2 = _mm_castsi128_ps( _mm_or_si128( z0, z1 ) );
	return array<float, 4>( z2 );
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

// XXX: use variadic template
template<class Result, int N, class T,  class Func>
Result call( Func const& f, array<T, N> arg0 ) {
	array<Result, N> result;
	for( int i = 0; i < N; ++i ) {
		result._data[i] = f( arg0._data[i] );
	}
}

template<class Container, class Elem>
struct container_view {
	container_view( Container* c ):
		_container( c ) {}

	template<class T, int N>
	reference<Elem, N> operator[]( array<T, N> const& idx ) const {
		reference<Elem, N> ref;
		for( int i = 0; i < N; ++i ) {
			ref._data[i] = &(*_container)[idx._data[i]];
		}
		return ref;
	}
	/*
	reference<Elem, 4> operator[]( array<int32_t, 4> const& idx ) const {
		union {
			int32_t array[4];
			__m128i packed;
		} y;
		y.packed = idx._packed;
		reference<Elem, 4> ref;
		ref._data[0] = &(*_container)[y.array[0]];
		ref._data[1] = &(*_container)[y.array[1]];
		ref._data[2] = &(*_container)[y.array[2]];
		ref._data[3] = &(*_container)[y.array[3]];
		return ref;
	}
	*/

	private:
		Container* _container;
};

template<class T>
container_view<T, typename T::value_type> view( T& c ) {
	return container_view<T, typename T::value_type>( &c );
}

typedef array<int32_t, 4> index;

struct Runner {
	template<class Func>
	void for_1d( int32_t x0, int32_t x1, Func const& f ) {
		index xs( 0, 1, 2, 3 );

		#pragma omp parallel for schedule(guided, 1)
		for( int32_t x = x0; x < x1; x += 4 ) {
			f( xs + x );
		}
	}

	template<class Func>
	void for_2d( int32_t x0, int32_t x1, int32_t y0, int32_t y1, Func const& f ) {
		index ys( 0, 0, 1, 1 );
		index xs( 0, 1, 0, 1 );
		#pragma omp parallel for schedule(guided, 1)
		for( int32_t y = y0; y < y1; y += 2 ) {
			index yi = ys + y;
			for( int32_t x = x0; x < x1; x += 2 ) {
				f( xs + x, yi );
			}
		}
	}
};


}
