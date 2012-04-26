#include <ostream>
#include <chrono>
#include <stdio.h>
#include "ssp.hpp"


/*
struct Particle {
	float qx, qy, qz;
	float px, py, pz;
	float m;
};

int main() {
	size_t const N = 16;
	ssp::vector<Particle> srcs( N );
	ssp::vector<float> dsts( N );

	ssp::Runner runner;
	
	runner.for_1d( 0, N, [&]( ssp::index const& i ) {
		auto px = srcs[i].member(&Particle::px);
		auto py = srcs[i].member(&Particle::py);
		auto pz = srcs[i].member(&Particle::pz);
		dsts[i] = sqrt( px * px + py * py + pz * pz );

		//auto mass = particle[i][&Particle::getMass]();
		//auto mass = particle[i].getMass();
	} );

	return 0;
}
*/

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

int const w = 4096, h = 4096;

void parallel( ssp::vector<int>& dst ) {
	using namespace ssp;


	Runner runner;
	runner.for_2d( 0, w, 0, h, [&]( index const& ix, index const& iy ) {
		array<float, 4> x = array<float, 4>( ix ) * (2.0f / w) - 1.0f;
		array<float, 4> y = array<float, 4>( iy ) * (2.0f / h) - 1.0f;
		array<float, 4> re = 0.0f;
		array<float, 4> im = 0.0f;
		array<float, 4> re2, im2;
		for( int i = 0; i < 256; ++i ) {
			re2 = re * re;
			im2 = im * im;
			if( all( re2 + im2 > 4.0f ) ) {
				break;
			}
			im = 2.0f * re * im + y;
			re = re2 - im2 + x;
		}
		// we must carefully treat NaN
		dst[ix + iy * w] = where<int32_t, 4>( re2 + im2 <= 4.0f, 0, 1 );
	} );
}

void serial( std::vector<int>& dst ) {
	for( int iy = 0; iy < h; ++iy ) {
		for( int ix = 0; ix < w; ++ix ) {
			float x = ix * (2.0f / w) - 1.0f;
			float y = iy * (2.0f / h) - 1.0f;
			float re = 0.0f;
			float im = 0.0f;
			float re2, im2;
			dst[ix + iy * w] = 0;
			for( int i = 0; i < 256; ++i ) {
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

int main() {
	using namespace ssp;

	std::vector<int32_t> dst_s( w * h );
	ssp::vector<int32_t> dst_p( w * h );

	auto sTimeBgn = std::chrono::high_resolution_clock::now();
	serial( dst_s );
	auto sTimeEnd = std::chrono::high_resolution_clock::now();
	printf( "%ld\n", (sTimeEnd - sTimeBgn).count() );

	auto pTimeBgn = std::chrono::high_resolution_clock::now();
	parallel( dst_p );
	auto pTimeEnd = std::chrono::high_resolution_clock::now();
	printf( "%ld\n", (pTimeEnd - pTimeBgn).count() );

	printf( "factor: %.2f\n",
		double( (sTimeEnd - sTimeBgn).count() ) / double( (pTimeEnd - pTimeBgn).count() )
	);

	int count = 0;
	for( size_t i = 0; i < w * h; ++i ) {
		if( dst_s[i] != dst_p[i] ) {
			++count;
		}
	}
	printf( "#error: %d\n", count );

	return 0;
}

