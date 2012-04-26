all:
	g++46 -g -std=c++0x -Wall -Wextra -O3 -msse4 -fopenmp test.cpp
