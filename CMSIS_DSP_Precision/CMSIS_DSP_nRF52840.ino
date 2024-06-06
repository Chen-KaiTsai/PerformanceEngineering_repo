/*
* @author ErebusTsai
* @note In most of my work, I use colSize, rowSize instead of nrow, ncol respectively.
* 
*/

#include "arm_math.h"
#include "Adafruit_TinyUSB.h"

const float32_t A_f32[4]{ 0.1, 0.2, 0.3, 0.4 };
q31_t A_q31[4];

const float32_t B_f32[16]{
  0.1, 0.2, 0.3, 0.4,
  0.5, 0.6, 0.7, 0.8,
  0.9, 0.1, 0.11, 0.12,
  0.13, 0.14, 0.15, 0.16
};
q31_t B_q31[16];

float32_t Y_f32[4];
q31_t Y_q31[4];
float32_t Y_f32q[4];

arm_status status;  // CMSIS-DSP calculation result status
arm_matrix_instance_f32 A;
arm_matrix_instance_f32 B;
arm_matrix_instance_f32 Y;

arm_matrix_instance_q31 Aq;
arm_matrix_instance_q31 Bq;
arm_matrix_instance_q31 Yq;

uint32_t srcRows, srcColumns;

void setup() {
  Serial.begin(9600);
  while (!Serial)
    ;
  Serial.print("XIAO-nRF52840 Start\n");

  // Initial f32 Matrix
  Serial.print("Start Initial f32 Matrix\n");

  srcRows = 1;
  srcColumns = 4;
  arm_mat_init_f32(&A, srcRows, srcColumns, (float32_t*)A_f32);

  srcRows = 4;
  srcColumns = 4;
  arm_mat_init_f32(&B, srcRows, srcColumns, (float32_t*)B_f32);

  srcRows = 1;
  srcColumns = 4;
  arm_mat_init_f32(&Y, srcRows, srcColumns, (float32_t*)Y_f32);

  // Initial q15 Matrix
  Serial.print("Start Initial q31 Matrix\n");

  arm_float_to_q31(A_f32, A_q31, 4);
  arm_float_to_q31(B_f32, B_q31, 16);

  srcRows = 1;
  srcColumns = 4;
  for (int i = 0; i < srcRows; ++i) {
    for (int j = 0; j < srcColumns; ++j) {
      Serial.print(A_q31[i * srcRows + j]);
      Serial.print(" ");
    }
    Serial.print("\n");
  }

  srcRows = 4;
  srcColumns = 4;
  for (int i = 0; i < srcRows; ++i) {
    for (int j = 0; j < srcColumns; ++j) {
      Serial.print(B_q31[i * srcRows + j]);
      Serial.print(" ");
    }
    Serial.print("\n");
  }

  srcRows = 1;
  srcColumns = 4;
  arm_mat_init_q31(&Aq, srcRows, srcColumns, (q31_t*)A_q31);

  srcRows = 4;
  srcColumns = 4;
  arm_mat_init_q31(&Bq, srcRows, srcColumns, (q31_t*)B_q31);
  
  srcRows = 1;
  srcColumns = 4;
  arm_mat_init_q31(&Yq, srcRows, srcColumns, (q31_t*)Y_q31);
}
void loop() {
  Serial.print("Start f32 matrix multiplication\n");
  status = arm_mat_mult_f32(&A, &B, &Y);

  if (status != ARM_MATH_SUCCESS) {
    printf("FAILURE\n");
    while (true)
      ;
  }

  Serial.print("Start q15 matrix multiplication\n");
  status = arm_mat_mult_q31(&Aq, &Bq, &Yq);

  if (status != ARM_MATH_SUCCESS) {
    printf("FAILURE\n");
    while (true)
      ;
  }

  Serial.print("Convert q15 output to f32\n");
  arm_q31_to_float(Yq.pData, Y_f32q, 4);

  for (int i = 0; i < srcRows; ++i) {
    for (int j = 0; j < srcColumns; ++j) {
      Serial.print(Y.pData[i * srcRows + j]);
      Serial.print(" ");
    }
    Serial.print("\n");
  }

  for (int i = 0; i < srcRows; ++i) {
    for (int j = 0; j < srcColumns; ++j) {
      Serial.print(Y_f32q[i * srcRows + j]);
      Serial.print(" ");
    }
    Serial.print("\n");
  }
}
