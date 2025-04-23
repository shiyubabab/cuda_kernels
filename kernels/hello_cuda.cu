#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hello_cude(){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("block id [%d], thread id [%d] and hello cuda!!!\n",blockIdx.x,idx);
}

int main(){
	hello_cuda<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}
