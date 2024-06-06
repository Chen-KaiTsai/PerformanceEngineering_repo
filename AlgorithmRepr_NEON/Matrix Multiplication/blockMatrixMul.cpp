/*
 * @author Chen-Kai Tsai
 * @date 06-06-2024
 * @note This arm neon implementation of blocked matrix multiplication also provide a padding to pad the input data to the multiple of the neon SIMD element size.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <arm_neon.h>

static void matrixMulF32_neon(float32_t* A, float32_t* B, float32_t* C, uint32_t H, uint32_t W, uint32_t inner) {
    int Aidx, Bidx, Cidx;
    float32x4_t A0, A1, A2, A3;
    float32x4_t B0, B1, B2, B3;
    float32x4_t C0, C1, C2, C3;

    for (unsigned int i = 0; i < H; i += 4) {
        for (unsigned int j = 0; j < W; j += 4) {
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (unsigned int k = 0; k < inner; k += 4) {
                Aidx = H * k + i;
                Bidx = inner * j + k;

                // Load 4x4 block from A
                A0 = vld1q_f32(A + Aidx);
                A1 = vld1q_f32(A + Aidx + H);
                A2 = vld1q_f32(A + Aidx + H * 2);
                A3 = vld1q_f32(A + Aidx + H * 3);

                B0 = vld1q_f32(B + Bidx);
                
                C0 = vmlaq_n_f32(C0, A0, B0[0]);
                C0 = vmlaq_n_f32(C0, A1, B0[1]);
                C0 = vmlaq_n_f32(C0, A2, B0[2]);
                C0 = vmlaq_n_f32(C0, A3, B0[3]);

                B1 = vld1q_f32(B + Bidx + inner);
                
                C1 = vmlaq_n_f32(C1, A0, B1[0]);
                C1 = vmlaq_n_f32(C1, A1, B1[1]);
                C1 = vmlaq_n_f32(C1, A2, B1[2]);
                C1 = vmlaq_n_f32(C1, A3, B1[3]);

                B2 = vld1q_f32(B + Bidx + inner * 2);
                
                C2 = vmlaq_n_f32(C2, A0, B2[0]);
                C2 = vmlaq_n_f32(C2, A1, B2[1]);
                C2 = vmlaq_n_f32(C2, A2, B2[2]);
                C2 = vmlaq_n_f32(C2, A3, B2[3]);

                B3 = vld1q_f32(B + Bidx + inner * 3);
                
                C3 = vmlaq_n_f32(C3, A0, B3[0]);
                C3 = vmlaq_n_f32(C3, A1, B3[1]);
                C3 = vmlaq_n_f32(C3, A2, B3[2]);
                C3 = vmlaq_n_f32(C3, A3, B3[3]);
            }

            Cidx = H * j + i;
            vst1q_f32(C + Cidx, C0);
            vst1q_f32(C + Cidx + H, C1);
            vst1q_f32(C + Cidx + H * 2, C2);
            vst1q_f32(C + Cidx + H * 3, C3);
        }
    }
}

static void matrixPadF32(float32_t* X, float32_t* Y, uint32_t H, uint32_t W, uint32_t blockSize) {
    if (Y != nullptr) {
        printf("Y is not null. Do nothing\n");
        return;
    }

    uint32_t y_h = ((H + blockSize - 1) / blockSize) * blockSize;
    uint32_t y_w = ((W + blockSize - 1) / blockSize) * blockSize;

    for (uint32_t i = 0; i < y_h; ++i) {
        for (uint32_t j = 0; j < y_w; ++j) {
            if (i >= H || j >= W)
                Y[i * y_w + j] = (float32_t)0.0f;
            else
                Y[i * y_w + j] = X[i * y_w + j];
        }
    }
}

int main(int argc, char** argv)
{
    int H = 8, W = 12, inner = 16;

    float32_t* A = (float32_t*)malloc(H * inner * sizeof(float32_t));
    float32_t* B = (float32_t*)malloc(W * inner * sizeof(float32_t));
    float32_t* C = (float32_t*)malloc(H * W * sizeof(float32_t));

    for (int i = 0; i < H * inner; i++) A[i]=(float32_t)rand()/(float32_t)RAND_MAX;
    for (int i = 0; i < W * inner; i++) B[i]=(float32_t)rand()/(float32_t)RAND_MAX;
    for (int i = 0; i < H * W; i++) C[i]=(float32_t)rand()/(float32_t)RAND_MAX;

    matrixMulF32_neon(A, B, C, H, W, inner);

    for (int i = 0; i < H * W; i++)
        printf("%5.5f,\t", C[i]);

    puts("\n");

    free(A);
    free(B);
    free(C);

    return 0;
}
