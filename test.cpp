#include <chrono>
#include <vector>
#include <ostream>
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>
#include <queue>
#include <tuple>
#include <stdio.h>
#include "ssp.hpp"
#include "ssp-math.hpp"


inline uint64_t rdtsc() {
	uint32_t lo, hi;
	__asm__ __volatile__( "rdtsc" : "=a"(lo), "=d"(hi) );
	return uint64_t( lo ) | (uint64_t( hi ) << 32);
}

template<class T, int N>
std::ostream& operator<<( std::ostream& out, ssp::array<T, N> const& arr ) {
	out << '[';
	if( N > 0 ) {
		out << arr._data[0];
		for( int i = 1; i < N; ++i ) {
			out << ", " << arr._data[i];
		}
	}
	out << ']';

	return out;
}

int const w = 512, h = 512;
//int const w = 8192, h = 8192;

template<class R, class View>
void test1_parallel( R& runner, View const& dst ) {
	using namespace ssp;

	runner.for_2d( 0, w, 0, h, [&]( typename R::a_int32 const& ix, typename R::a_int32 const& iy ) {
		typename R::a_float x = typename R::a_float( ix ) * (2.0f / w) - I;
		typename R::a_float y = typename R::a_float( iy ) * (2.0f / h) - I;
		typename R::a_float re = O;
		typename R::a_float im = O;
		typename R::a_float re2, im2;
		for( int i = 0; i < 1; ++i ) {
			re2 = re * re;
			im2 = im * im;
			if( all_of( re2 + im2 > 4.0f ) ) {
				break;
			}
			im = 2.0f * re * im + y;
			re = re2 - im2 + x;
		}
		// we must treat NaN carefully, which comes from (inf - inf).
		dst[ix + iy * w] = where( re2 + im2 <= 4.0f, typename R::a_int32( 0 ), 1 );
	} );
}

void test1_serial( std::vector<int>& dst ) {
	for( int iy = 0; iy < h; ++iy ) {
		for( int ix = 0; ix < w; ++ix ) {
			float x = ix * (2.0f / w) - 1.0f;
			float y = iy * (2.0f / h) - 1.0f;
			float re = 0.0f;
			float im = 0.0f;
			float re2, im2;
			dst[ix + iy * w] = 0;
			for( int i = 0; i < 1; ++i ) {
				re2 = re * re;
				im2 = im * im;
				if( re2 + im2 > 4.0f ) {
					dst[ix + iy * w] = 1;
					break;
				}
				im = 2.0f * re * im + y;
				re = re2 - im2 + x;
			}
		}
	}
}

template<class Vector>
Vector factorial_parallel( Vector const& n ) {
	if( all_of( n <= 1 ) ) {
		return 1;
	}
	return where( n <= 1,
		1,
		n * factorial_parallel( n - 1 )
	);
}

template<class Vector>
Vector factorial_serial( Vector const& n ) {
	if( n <= 1 ) {
		return 1;
	}
	else {
		return n * factorial_serial( n - 1 );
	}
}

struct Vec2 {
	float x, y;

	bool operator==( Vec2 const& rhs ) {
		return x == rhs.x && y == rhs.y;
	}

	bool operator!=( Vec2 const& rhs ) {
		return !(*this == rhs);
	}
};

/*
template<class Array>
struct ssp_traits<Vec2> {
	ssp_traits<Vec2>( Array* self ):
		x( self ),
		y( self ) {}
	define_member<Array, &Vec2::x> x;
	define_member<Array, &Vec2::y> y;
};
*/

void test0_parallel( std::vector<Vec2>& srcv, std::vector<Vec2>& dstv ) {
	auto srcs = ssp::const_view( srcv );
	auto dsts = ssp::view( dstv );

	typedef ssp::Runner<4, 0> R;
	R runner;
	
	runner.for_1d( 0, srcv.size(), [&]( R::a_int32 const& i ) {
		R::a_float x = srcs[i].member( &Vec2::x );
		R::a_float y = srcs[i].member( &Vec2::y );
		/*
		ssp::array<float, 4> y = srcs[i].memfun( &Vec2::getPos )();
		ssp::array<float, 4> y = call( bind( &Vec2::getPos ), srcs[i] );
		*/
		dsts[i].member( &Vec2::x ) = x + y;
		dsts[i].member( &Vec2::y ) = x - y;
	} );
}

void test0_parallel_opt( std::vector<Vec2>& srcv, std::vector<Vec2>& dstv ) {
	ssp::Runner<4, 0> runner;
	
	runner.for_1d( 0, srcv.size(), [&]( ssp::Runner<4, 0>::a_int32 const& i ) {
		ssp::array<float, 4> x(
			srcv[i._data[0]].x,
			srcv[i._data[1]].x,
			srcv[i._data[2]].x,
			srcv[i._data[3]].x
		);
		ssp::array<float, 4> y(
			srcv[i._data[0]].y,
			srcv[i._data[1]].y,
			srcv[i._data[2]].y,
			srcv[i._data[3]].y
		);
		ssp::array<float, 4> u = x + y;
		ssp::array<float, 4> v = x - y;
		dstv[i._data[0]].x = u._data[0];
		dstv[i._data[1]].x = u._data[1];
		dstv[i._data[2]].x = u._data[2];
		dstv[i._data[3]].x = u._data[3];
		dstv[i._data[0]].y = v._data[0];
		dstv[i._data[1]].y = v._data[1];
		dstv[i._data[2]].y = v._data[2];
		dstv[i._data[3]].y = v._data[3];
	} );
}

void test0_parallel_opt2( std::vector<Vec2>& srcv, std::vector<Vec2>& dstv ) {
	for( size_t i = 0; i < srcv.size(); i += 4 ) {
		ssp::array<float, 4> x(
			srcv[i + 0].x,
			srcv[i + 1].x,
			srcv[i + 2].x,
			srcv[i + 3].x
		);
		ssp::array<float, 4> y(
			srcv[i + 0].y,
			srcv[i + 1].y,
			srcv[i + 2].y,
			srcv[i + 3].y
		);
		ssp::array<float, 4> u = x + y;
		ssp::array<float, 4> v = x - y;
		dstv[i + 0].x = u._data[0];
		dstv[i + 1].x = u._data[1];
		dstv[i + 2].x = u._data[2];
		dstv[i + 3].x = u._data[3];
		dstv[i + 0].y = v._data[0];
		dstv[i + 1].y = v._data[1];
		dstv[i + 2].y = v._data[2];
		dstv[i + 3].y = v._data[3];
	}
}
void test0_serial( std::vector<Vec2> const& srcv, std::vector<Vec2>& dstv ) {
	for( size_t i = 0; i < srcv.size(); ++i ) {
		float x = srcv[i].x;
		float y = srcv[i].y;
		dstv[i].x = x + y;
		dstv[i].y = x - y;
	}
}

void test0() {
	using namespace std;

	size_t const N = 256 * 1024; // larger than L2 cache
	size_t const M = 1024;
	std::vector<Vec2> src( N );
	std::vector<Vec2> dst_s( N );
	std::vector<Vec2> dst_p( N );

	auto pTickBgn = chrono::high_resolution_clock::now();
	for( size_t i = 0; i < M; ++i ) {
		test0_parallel( src, dst_p );
	}
	auto pTickEnd = chrono::high_resolution_clock::now();
	printf( "parallel: %lu\n", (pTickEnd - pTickBgn).count() );

	auto sTickBgn = chrono::high_resolution_clock::now();
	for( size_t i = 0; i < M; ++i ) {
		test0_serial( src, dst_s );
	}
	auto sTickEnd = chrono::high_resolution_clock::now();
	printf( "  serial: %lu\n", (sTickEnd - sTickBgn).count() );

	printf( "factor: %.2f\n",
		double( (sTickEnd - sTickBgn).count() ) / double( (pTickEnd - pTickBgn).count() )
	);

	int count = 0;
	for( size_t i = 0; i < N; ++i ) {
		if( dst_s[i] != dst_p[i] ) {
			++count;
		}
	}
	printf( "#error: %d\n", count );
}

int32_t test( int32_t x ) {
	return x;
}

template<class Rng>
float hidoi_float( int n, Rng& rng ) {
	using namespace std;

	static uint32_t const mask = 0xff << 23;
	uniform_int_distribution<int> fdist( 0, n + 0x7f );

	union {
		float f;
		uint32_t b;
	} v;
	v.b = (uint32_t( rng() ) & ~mask) | (fdist( rng ) << 23);
	return v.f;
}

template<class F0, class F1>
void test_math_func( double m, size_t N, F0 f0, F1 f1 ) {
	using namespace std;

	priority_queue<tuple<double, double>> errors;
	for( int i = 0; i < 4; ++i ) {
		errors.push( make_tuple( 0.0, 0.0 ) );
	}

	int n = log( m ) / log( 2 );

	mt19937 rng;
	for( size_t i = 0; i < N; ++i ) {
		float x = hidoi_float( n, rng );
		double r0 = f0( ssp::array<float, 4>( x ) )._data[0];
		double r1 = f1( x );
		if( r0 != r0 && r1 != r1 ) {
			continue;
		}
		if( r0 == 0.0 && r1 == 0.0 && signbit( r0 ) == signbit( r1 ) ) {
			continue;
		}
		if( r0 == r0 && r1 == r1 ) {
			r0 = min<double>( max<double>( r0, -numeric_limits<float>::max() ), +numeric_limits<float>::max() );
			r1 = min<double>( max<double>( r1, -numeric_limits<float>::max() ), +numeric_limits<float>::max() );
			double err = fabs( r0 - r1 ) / max<double>( fabs( r1 ), numeric_limits<float>::min() );
			errors.push( make_tuple( -err, x ) );
			errors.pop();
			continue;
		}
		printf( "error at x = %+e\n", x );
	}
	while( !errors.empty() ) {
		double err, x;
		tie( err, x ) = errors.top();
		printf( "error %e at x = %+e\n", -err, x );
		errors.pop();
	}
	printf( "\n" );
}

template<class Func>
int64_t benchmark( Func const& f ) {
	auto tickBgn = std::chrono::high_resolution_clock::now();
	f();
	auto tickEnd = std::chrono::high_resolution_clock::now();
	return (tickEnd - tickBgn).count();
}

#include <unistd.h>

int main() {
	using namespace ssp;

	int64_t t = benchmark( []() {
		size_t const N = 1ul << 24;
		printf( "floor:\n" );
		test_math_func( 1e9f, N, &ssp::floor, (double (*)(double))std::floor );
		printf( "ceil:\n" );
		test_math_func( 1e9f, N, &ssp::ceil, (double (*)(double))std::ceil );
		printf( "sin:\n" );
		test_math_func( 256.0f, N, &ssp::sin<4>, (double (*)(double))std::sin );
		printf( "cos:\n" );
		test_math_func( 256.0f, N, &ssp::cos<4>, (double (*)(double))std::cos );
		printf( "tan:\n" );
		test_math_func( 256.0f, N, &ssp::tan<4>, (double (*)(double))std::tan );
		printf( "asin:\n" );
		test_math_func( 2.0f, N, &ssp::asin<4>, (double (*)(double))std::asin );
		printf( "acos:\n" );
		test_math_func( 2.0f, N, &ssp::acos<4>, (double (*)(double))std::acos );
		printf( "atan:\n" );
		test_math_func( 1e3f, N, &ssp::atan<4>, (double (*)(double))std::atan );
		printf( "sinh:\n" );
		test_math_func( 2.0f, N, &ssp::sinh<4>, (double (*)(double))std::sinh );
		printf( "cosh:\n" );
		test_math_func( 2.0f, N, &ssp::cosh<4>, (double (*)(double))std::cosh );
		printf( "tanh:\n" );
		test_math_func( 63.0f, N, &ssp::tanh<4>, (double (*)(double))std::tanh );
		printf( "exp:\n" );
		test_math_func( 63.0f, N, &ssp::exp<4>, (double (*)(double))std::exp );
		printf( "log:\n" );
		test_math_func( 1e30f, N, &ssp::log<4>, (double (*)(double))std::log );
	} );

	std::cout << t << std::endl;


	std::vector<int32_t> dst_s( w * h );
	std::vector<int32_t> dst_p( w * h );

	sleep( 1 );

	auto sTickBgn = std::chrono::high_resolution_clock::now();
	for( int i = 0; i < 1024; ++i ) {
		test1_serial( dst_s );
	}
	auto sTickEnd = std::chrono::high_resolution_clock::now();
	printf( "  serial: %lu\n", (sTickEnd - sTickBgn).count() );

	Runner<4, 0> runner;
	auto pTickBgn = std::chrono::high_resolution_clock::now();
	for( int i = 0; i < 1024; ++i ) {
		test1_parallel( runner, view( dst_p ) );
	}
	auto pTickEnd = std::chrono::high_resolution_clock::now();
	printf( "parallel: %lu\n", (pTickEnd - pTickBgn).count() );

	printf( "factor: %.2f\n",
		double( (sTickEnd - sTickBgn).count() ) / double( (pTickEnd - pTickBgn).count() )
	);

	int count = 0;
	for( size_t i = 0; i < w * h; ++i ) {
		if( dst_s[i] != dst_p[i] ) {
			++count;
		}
	}
	printf( "#error: %d\n", count );

	//test0();

	return 0;
}
