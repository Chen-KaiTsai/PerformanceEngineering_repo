#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>

#include <immintrin.h> //AVX, AVX2, FMA

#include "malloc2D.h"

void convolution2D(int inputH, int inputW, int inChannel, int outputH, int outputW, int outChannel, int batchSize, int stride, int kSize, int padSize, float* X, float* W, float* B, float* Y);
void convolution2D_Iavx(int inputH, int inputW, int inChannel, int outputH, int outputW, int outChannel, int batchSize, int stride, int kSize, int padSize, float* X, float* W, float* B, float* Y);
