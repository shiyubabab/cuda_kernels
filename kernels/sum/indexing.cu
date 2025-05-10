#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef int data_t;
const int N = 32;

__global__ void sum(data_t *data){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	printf("current thread global id is %d\n,and in thread block index is %d",idx,tx);

	data[idx] += 1;
}

int main(void){
	int nbytes = sizeof(data_t) * N;
	data_t *d_mem,*h_mem;
	h_mem = (data_t *)malloc(nbytes);
	cudaMalloc((void **)&d_mem,nbytes);

	for(int i = 0 ;i<N;i++){
		h_mem[i] = i;
		printf("%d\n",h_mem[i]);
	}

	cudaMemcpy(d_mem,h_mem,nbytes,cudaMemcpyHostToDevice);

	sum<<<1,N>>>(d_mem);

	cudaMemcpy(h_mem,d_mem,nbytes,cudaMemcpyDeviceToHost);

	for(int i =0;i<N;i++) printf("%d\n",h_mem);
	cudaFree(d_mem);

	return 0;
}
