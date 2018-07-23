////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <time.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

//void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size);


namespace FengML
{
	template<class T>
	void matrixMul_CPU(T *A, int hA ,int wA, T *B, int hB,  int wB, T *C);

	template<class T>
	void matrixMul_CPU(T *A, int hA ,int wA, T *B, int hB,  int wB, T *C)
	{
		memset(C, 0.0, hA * wB * sizeof(float));
		unsigned int i, j, k;
		for (i = 0; i < hA; ++i)
			for (j = 0; j < wB; ++j)
			{
				for (k = 0; k < wA; ++k)
				{
					C[i * wB + j] += A[i * wA + k] * B[k * wB + j];
				}
				 
			}
	}

	template<class T>
	void matrixAdd_CPU(T *A, int hA, int wA, T *B, int hB, int wB, T *C);

	template<class T>
	void matrixAdd_CPU(T *A, int hA, int wA, T *B, int hB, int wB, T *C)
	{
		for (unsigned int i = 0; i < hA; ++i)
			for (unsigned int j = 0; j < wB; ++j)
			{
				T sum = 0;

				for (unsigned int k = 0; k < wA; ++k)
				{
					T a = A[i * wA + k];
					T b = B[k * wB + j];
					sum += a * b;
				}

				C[i * wB + j] += (T)sum;
			}
	}
		template<class T>
		//C = A*B
		void matrixMul_GPU(T* A, int A_h, int A_w, T* B, int B_h, int B_w, T* C);

		template<class T>
		//C = A*B
		void matrixMul_GPU(T* A, int A_h, int A_w, T* B, int B_h, int B_w, T* C)
		{
			cudaDeviceProp deviceProp;
			int devID = 0;
			checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));



			unsigned int size_A = A_h * A_w;
			unsigned int mem_size_A = sizeof(T) * size_A;

			unsigned int size_B = B_h * B_w;
			unsigned int mem_size_B = sizeof(T) * size_B;

			unsigned int size_C = A_h * B_w;
			unsigned int mem_size_C = sizeof(T) * size_C;

			// allocate device memory 在显卡上开辟空间
			//要将内存的数据放到显卡里面
			T *d_A, *d_B, *d_C;

			cudaEvent_t all_start, all_stop;
			checkCudaErrors(cudaEventCreate(&all_start));
			checkCudaErrors(cudaEventCreate(&all_stop));
			// allocate host memory for the result
			checkCudaErrors(cudaEventRecord(all_start, NULL));
			checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
			checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));

			checkCudaErrors(cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaEventRecord(all_stop, NULL));
			checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

			// CUBLAS version 2.0
			{
				const T alpha = 1.0f;
				const T beta = 0.0f;
				cublasHandle_t handle;
				cudaEvent_t start, stop;
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				checkCudaErrors(cublasCreate(&handle));

				checkCudaErrors(cudaEventRecord(start, NULL));
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_w, A_h, A_w, &alpha, d_B, B_w, d_A, A_w, &beta, d_C, B_w));

				// Record the stop event
				checkCudaErrors(cudaEventRecord(stop, NULL));

				// Wait for the stop event to complete
				checkCudaErrors(cudaEventSynchronize(stop));
				checkCudaErrors(cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

				float msecTotal = 0.0f;
				float mescAll = 0.0f;
				checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
				checkCudaErrors(cudaEventElapsedTime(&mescAll, all_start, all_stop));


				// Compute and print the performance
				float msecPerMatrixMul = msecTotal ;
				double flopsPerMatrixMul = 2.0 * (double)A_h * (double)A_w * (double)B_w;
				double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

				printf(
					"Performance= %.2f GFlop/s, Time= %.3f msec, All time = %.3f msec, vSize= %.0f Ops\n",
					gigaFlops,
					msecPerMatrixMul,
					mescAll,
					flopsPerMatrixMul);
				// Destroy the handle
				checkCudaErrors(cublasDestroy(handle));
			}

			// clean up memory
			checkCudaErrors(cudaFree(d_A));
			checkCudaErrors(cudaFree(d_B));
			checkCudaErrors(cudaFree(d_C));
		}

		template<class T>
		//C = A*B + C
		void matrixAdd_GPU(T* A, int A_h, int A_w, T* B, int B_h, int B_w, T* C); //A*B+C

		template<class T>
		//C = A*B + C  这个函数主要用作矩阵加法
		void matrixAdd_GPU(T* A, int A_h, int A_w, T* B, int B_h, int B_w, T* C)
		{
			cudaDeviceProp deviceProp;
			int devID = 0;
			checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));



			unsigned int size_A = A_h * A_w;
			unsigned int mem_size_A = sizeof(T) * size_A;

			unsigned int size_B = B_h * B_w;
			unsigned int mem_size_B = sizeof(T) * size_B;

			unsigned int size_C = A_h * B_w;
			unsigned int mem_size_C = sizeof(T) * size_C;

			// allocate device memory 在显卡上开辟空间
			//要将内存的数据放到显卡里面
			T *d_A, *d_B, *d_C;


			// allocate host memory for the result
			checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
			checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
			checkCudaErrors(cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice));

			checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));
			checkCudaErrors(cudaMemcpy(d_C, C, mem_size_C, cudaMemcpyHostToDevice));

			// CUBLAS version 2.0
			{
				const T alpha = 1.0f;
				const T beta = 1.0f;
				cublasHandle_t handle;

				checkCudaErrors(cublasCreate(&handle));

				// Allocate CUDA events that we'll use for timing

				//	checkCudaErrors(cudaEventCreate(&stop));



				// Record the stop event


				//note cublas is column primary!
				//need to transpose the order
				//			checkCudaErrors(cudaEventCreate(&stop));
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_w, A_h, A_w, &alpha, d_B, B_w, d_A, A_w, &beta, d_C, B_w));
				//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

				//			checkCudaErrors(cudaEventRecord(stop, NULL));
				// Wait for the stop event to complete
				//		checkCudaErrors(cudaEventRecord(stop, NULL));

				// Wait for the stop event to complete
				//checkCudaErrors(cudaEventSynchronize(stop));

				// copy result from device to host				checkCudaErrors(cudaMemcpy(H_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
				// Destroy the handle
				checkCudaErrors(cublasDestroy(handle));
			}

			// clean up memory
			checkCudaErrors(cudaFree(d_A));
			checkCudaErrors(cudaFree(d_B));
			checkCudaErrors(cudaFree(d_C));
		}
}
