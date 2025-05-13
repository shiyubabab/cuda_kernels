#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int blockSize>
__global__ void matrix_transpose_v0(const float *input,float *output,int X,int Y){
	const int ty = threadIdx.y,tx = threadIdx.x;

	int gtx = blockIdx.x * blockSize + tx;
	int gty = blockIdx.y * blockSize + ty;
	__shared__ float sdata[blockSize][blockSize];

	if(gty<Y && gtx<X){
		sdata[ty][tx] = input[gty * X + gtx];
	}
	__syncthreads();

	gtx = blockIdx.y * blockSize + tx;
	gty = blockIdx.x * blockSize + ty;
	if(gty<Y && gtx<X){
		output[gty * Y + gtx] = sdata[tx][ty];
	}
}


int main(void){
	float milliseconds = 0;
	const int Y = 2300;
	const int X = 1500;
	constexpr int N = X*Y;
	int matrix_size = N * sizeof(float);
	float *h_input,*d_input;
	float *h_output,*d_output;
	h_input = (float *)malloc(matrix_size);
	for(int i = 0;i<N;i++){
		h_input[i] = 2.0f;
	}
	cudaMalloc((void **)&d_input,matrix_size);
	cudaMemcpy(d_input,h_input,matrix_size,cudaMemcpyHostToDevice);

	h_output = (float *)malloc(matrix_size);
	cudaMalloc((void **)&d_output,matrix_size);
	
	const int blockSize = 16;
	int gridSize = (N+blockSize-1)/blockSize;
	dim3 block(blockSize,blockSize);
	dim3 grid(gridSize,gridSize);
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	matrix_transpose_v0<blockSize><<<grid,block>>>(d_input,d_output,X,Y);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);

	cudaMemcpy(h_output,d_output,matrix_size,cudaMemcpyDeviceToHost);
	printf("matrix transpose latency = %f ms \n",milliseconds);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);
	return 0;
}
