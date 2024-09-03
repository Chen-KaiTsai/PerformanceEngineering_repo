#include "main.h"

int main(int argc, char** argv)
{
    // S == inner
	constexpr int N = 1024, M = 1024, S = 1024;

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

    auto start = std::chrono::steady_clock::now();

    matrixMulF32_VCL2(c, a, b, N, M, S);

    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Compute Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);

#ifdef DEBUG
	for (int i = 0; i < N * M; ++i) {
		if (i % M == 0) {
			puts("");
		}
		printf("%3.1f\t", c[i]);
	}
    puts("\n");
#endif
}
