// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;

typedef union _avxf {
    __m256 val;
    float arr[8];
} avxf;





Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */

    // Create a matrix to store the results
    Matrix<float> result(a.rows, b.columns);

    //transpose matrix b
    Matrix<float> b_t(b.columns, b.rows);
    for(int i = 0; i < b.rows; i++){
        for(int j = 0; j < b.columns; j++){
            b_t(j, i) = b(i, j);
        }
    }




    //use simd for fast matrix multiplication
    __m256 a_row, b_col, sum;
    _avxf sum_avx;
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b_t.rows; j++){
            sum = _mm256_setzero_ps();
            for(int k = 0; k < a.columns; k+=8)
                   if(k+8 > a.columns) {
                       //temp sum
                       float temp_sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                       for(int l = 0; l < a.columns - k; l++){
                           temp_sum[l] = a(i, k+l) * b_t(j, k+l);
                       }
                          sum = _mm256_add_ps(sum, _mm256_loadu_ps(temp_sum));
                   }else {

                       a_row = _mm256_loadu_ps(&a(i, k));
                       b_col = _mm256_loadu_ps(&b_t(j, k));
                       sum = _mm256_add_ps(sum, _mm256_mul_ps(a_row, b_col));
                   }
            //use horizontal add to sum up the 8 elements in sum
            sum=_mm256_hadd_ps(sum, sum);
            sum=_mm256_hadd_ps(sum, sum);
            //store sum in result(i, j)
            _mm256_storeu_ps(sum_avx.arr, sum);
            result(i, j) = sum_avx.arr[0] + sum_avx.arr[4];
        }
    }
    return result;
}


Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  // Create a matrix to store the results
    Matrix<double> result(a.rows, b.columns);
    //Create a vector to store the results of a row of a and a column of b
    //transpose matrix b
    Matrix<double> b_t(b.columns, b.rows);
    for(int i = 0; i < b.rows; i++){
        for(int j = 0; j < b.columns; j++){
            b_t(j, i) = b(i, j);
        }
    }

    //use simd for fast matrix multiplication
    __m256d a_row, b_col, sum;
    _avxd sum_avx;
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b_t.rows; j++){
            sum = _mm256_setzero_pd();
            for(int k = 0; k < a.columns; k+=4)
                   if(k+4 > a.columns) {
                       //temp sum
                       double temp_sum[4] = {0, 0, 0, 0};
                       for(int l = 0; l < a.columns - k; l++){
                           temp_sum[l] = a(i, k+l) * b_t(j, k+l);
                       }
                          sum = _mm256_add_pd(sum, _mm256_loadu_pd(temp_sum));
                   }else {



                       a_row = _mm256_loadu_pd(&a(i, k));
                       b_col = _mm256_loadu_pd(&b_t(j, k));
                       sum = _mm256_add_pd(sum, _mm256_mul_pd(a_row, b_col));
                   }
            //use horizontal add to sum up the 4 elements in sum
            sum=_mm256_hadd_pd(sum, sum);
            _mm256_storeu_pd(sum_avx.arr, sum);
            result(i, j) = sum_avx.arr[0]+sum_avx.arr[2];
        }
    }
    return result;

}
/*************************************/
#pragma GCC pop_options