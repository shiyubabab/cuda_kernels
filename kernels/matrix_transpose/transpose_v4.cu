#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FETCH_CFLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])
#define FETCH_FLOAT4(p) (reinterpret_cast<float4*>(&(p))[0])

template<int blockSize>
__global__ void matrix_transpose_v0(const float *input,float *output,int X,int Y){
	const int ty = threadIdx.y,tx = threadIdx.x;

	__shared__ float sdata[blockSize][blockSize];

	int gtx = blockIdx.x * blockSize + tx;
	int gty = blockIdx.y * blockSize + ty;

	if(gty<Y && gtx<X){
		FETCH_FLOAT4(sdata[ty][tx * 4]) = FETCH_FLOAT4(input[gty * Y + gtx]);
	}
	__syncthreads();

	gtx = blockIdx.y * blockSize + tx;
	gty = blockIdx.x * blockSize + ty;
	float tmp[4];
	if(gty < X && gtx < Y){
		for(int i = 0;i<4;i++){
			tmp[i] = sdata[tx*4+i][ty];
		}
		FETCH_FLOAT4(output[gty*Y+gtx]) = FETCH_FLOAT4(tmp);
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
	
	const int blockSize = 32;
	int gridSize = (N+blockSize-1)/blockSize;
	dim3 block(blockSize/4,blockSize);
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
