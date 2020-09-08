CXXFLAGS = -Wno-unused-variable -Wall -Wextra -pedantic -std=gnu++17 -g -O3
CFLAGS = -std=c11

CXX = g++
CC = gcc


all: lib.cc main.cc
	$(CXX) $(CXXFLAGS) *.h *.cc -o main

clean:
	-rm *.o *.out

