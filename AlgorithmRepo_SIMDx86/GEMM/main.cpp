#include "main.h"

int main(int argc, char** argv)
{
	int N = 16, M = 8, S = 5;

	float** a_data = (float**)malloc2Df(N, S);
	float** b_data = (float**)malloc2Df(S, M);
	float** c_data = (float**)malloc2Df(N, M);

	float* a = a_data[0];
	float* b = b_data[0];
	float* c = c_data[0];

	for (int i = 0; i < N * S; ++i)
		a[i] = 1.0f;
	for (int i = 0; i < S * M; ++i)
		b[i] = 1.0f;

	matrixMulF32_Iavx2(c, a, b, N, M, S);

	for (int i = 0; i < N * M; ++i) {
		if (i % M == 0) {
			puts("");
		}
		printf("%3.1f\t", c[i]);
	}

#ifdef TRANSPOSE_TEST
	int N = 8, M = 8;

	float** a_data = (float**)malloc2Df(N , M);
	float** b_data = (float**)malloc2Df(M , N);

	float* a = a_data[0];
	float* b = b_data[0];

	for (int i = 0; i < N * M; ++i)
		a[i] = static_cast<float>(i);
	
	for (int i = 0; i < N * M; ++i) {
		if (i % M == 0) {
			puts("");
		}
		printf("%3.1f\t", a[i]);
	}

	puts("\n\n");

	matrixTransposeBlocks(a, b, N, M, 8);

	for (int i = 0; i < N * M; ++i) {
		if (i % N == 0) {
			puts("");
		}
		printf("%3.1f\t", b[i]);
	}
	
	free(a_data);
	free(b_data);
#endif
	return 0;
}
