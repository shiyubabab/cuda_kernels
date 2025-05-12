#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int blockSize>
__global__ void reduce_v0(const float *input,float *output){
	int tid = threadIdx.x;
	__shared__ float sdata[blockSize];

	sdata[tid] = input[tid];
	__syncthreads();

	for(int s = 1; s < blockSize;s *= 2){
		if(tid% (s * 2)==0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid == 0){
		output[blockIdx.x] = sdata[tid];
	}
}

int main(void){
	const int N = 25600000;
	float milliseconds = 0;

	const int blockSize = 256;
	int GridSize = (N + blockSize -1)/blockSize;
	float *h_mem,*d_mem;
	h_mem = (float *) malloc(sizeof(float)*N);
	cudaMalloc((void **)&d_mem,sizeof(float)*N);

	float *h_ret,*d_ret;
	h_ret = (float *)malloc(sizeof(float)*GridSize);
	cudaMalloc((void **)&d_ret,sizeof(float)*GridSize);

	for(int i = 0 ;i<N ;i++) h_mem[i] = 1.0f;

	cudaMemcpy(d_mem,h_mem,sizeof(float)*N,cudaMemcpyHostToDevice);

	dim3 Grid(GridSize);
	dim3 Block(blockSize);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	reduce_v0<blockSize><<<Grid,Block>>>(d_mem,d_ret);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds,start,stop);

	cudaMemcpy(h_ret,d_ret,GridSize*sizeof(float),cudaMemcpyDeviceToHost);
	float res = 0;
	for(int i = 0; i <GridSize;i++){
		res += h_ret[i];
	}
	printf("The result is %f \n",res);
	printf("The reduce_v0 latency = %f ms \n",milliseconds);

	cudaFree(d_mem);
	cudaFree(d_ret);
	free(h_mem);
	free(h_ret);
	return 0;
}
