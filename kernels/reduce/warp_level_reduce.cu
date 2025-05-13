#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float WarpShuffle(float sum){
	sum += __shlf_down_sync(0xffffffff,sum,16);
	sum += __shlf_down_sync(0xffffffff,sum,8);
	sum += __shlf_down_sync(0xffffffff,sum,4);
	sum += __shlf_down_sync(0xffffffff,sum,2);
	sum += __shlf_down_sync(0xffffffff,sum,1);
	return sum;
}

template<int blockSize>
__global__ void reduce_v6(const float *input,float *output,unsigned int n){
	float sum = 0;

	int tid = threadIdx.x;
	int gtid = threadIdx.x + blockIdx.x * blockSize;

	unsigned int total_thread_num = blockSize * gridDim.x;

	for(int i = gtid;i<n;i+=total_thread_num){
		sum += input[i];
	}

	__shared__ float warpsums[blockSize/32];
	const int laneId = tid % 32;
	const int warpId = tid / 32;
	sum = WarpShuffle(sum);
	if(laneId == 0){
		warpsums[warpId] = sum;
	}
	__syncthreads();

	sum = (tid<blockSize/32) ? warpsums[laneId] : 0;
	if(warpId == 0){
		sum = WarpShffle(sum);
	}

	if(tid == 0){
		output[blockIdx.x] = sum;
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
	reduce_v6<blockSize><<<Grid,Block>>>(d_mem,d_ret);
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
