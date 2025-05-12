#include <stdio.h>
#include <cuda.h>
#include <cudaruntime.h>

int cpu_filter_baseline(int *dst,const int *src,int n){
	int nres = 0;
	for(int i = 0;i<n;i++){
		if(src[i]>0){
			dst[nres++] = src[i];
		}
	}
	return nres;
}

int filter_baseline(int *dst,int *nres,const int *src,int n){
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(src[tid] > 0 && tid < n) dst[atomicAdd(nres,1)] = src[tid];
}

int block_filter(int *dst,int *nres,const int *src,int n){
	unsigned int tid  = threadIdx.x;
	unsigned int gtid = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int total_thread_num = gridDim.x*blockDim.x;

	__shared__ int snres;
	for(int i = gtid; i<n ;i+=total_thread_num){
		if(tid == 0){
			snres = 0;
		}
		__syncthreads();
		int d,pos;
		d = src[i];
		if(i<n && d>0){
			pos = atomicAdd(&snres,1);
		}
		__syncthreads();

		if(tid == 0) snres = atomicAdd(nres,snres);
		__syncthreads();

		if(i<n && d>0){
			pos += snres;
			dst[pos] = d;
		}
		__syncthreads();
	}

}

bool CheckResult(int *out,int groudtruth){
	if(*out != groudtruth) return false;
	return true;
}

int main(void){
	float milliseconds = 0;
	int N = 2560000;

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	const int blockSize = 256;
	int GridSize = (N + 256 - 1) / 256;

	int *src_h = (int *)malloc(N * sizeof(int));
	int *dst_h = (int *)malloc(N * sizeof(int));
	int *nres_h = (int *)malloc(1 * sizeof(int));
	int *dst, *nres;
	int *src;
	cudaMalloc((void **)&src, N * sizeof(int));
	cudaMalloc((void **)&dst, N * sizeof(int));
	cudaMalloc((void **)&nres, 1 * sizeof(int));
	for(int i = 0; i < N; i++){
		src_h[i] = 1;
	}
	int groudtruth = 0;
	for(int j = 0; j < N; j++){
		if (src_h[j] > 0) {
			groudtruth += 1;
		}

	cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid(GridSize);
	dim3 Block(blockSize);
																					    
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	block_filter<<<Grid, Block>>>(dst, nres, src, N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	bool is_right = CheckResult(nres_h, groudtruth, N);
	if(is_right) {
		printf("the ans is right\n");
	} else {
		printf("the ans is wrong\n");
		printf("%lf ",*nres_h);
		printf("\n");
	}
	printf("block_filter latency = %f ms\n", milliseconds);    
	cudaFree(src);
	cudaFree(dst);
	cudaFree(nres);
	free(src_h);
	free(dst_h);
	free(nres_h);
	return 0;
}
