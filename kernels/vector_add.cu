#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MEMORY_OFFSET 10000000
typedef float fsize;

__global__ void vec_add(fsize *x,fsize *y,fsize *z,int N){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < N) z[idx] = x[idx] + y[idx];
}

__global__ void vec_add_float4(float *A,float *B,float *C){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_float4 = MEMORY_OFFSET / 4;
	int num_remaining_floats = MEMORY_OFFSET % 4;
	for(int i = idx;i<num_float4;i+=blockDim.x * gridDim.x){
		float4 a1 = reinterpret_cast<float4*>(A)[i];
		float4 b1 = reinterpret_cast<float4*>(B)[i];
		float4 c1;

		c1 = a1.x + b1.x;
		c1 = a1.y + b1.y;
		c1 = a1.z + b1.z;
		c1 = a1.w + b1.w;

		reinterpret_cast<float4>(C)[i] = c1;
	}

	if(idx < num_remaining_floats){
		int remaining_start_index = num_float4 * 4;
		C[remaining_start_index + idx] = A[remaining_start_index + idx] + B[remaining_start_index + idx];
	}

}

void vec_add_cpu(fsize *x,fsize *y,fsize *z,int N){
	for(int i = 0;i<N;i++) z[i] = y[i] + x[i];
}
