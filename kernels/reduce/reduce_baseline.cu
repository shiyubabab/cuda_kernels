#include <stdio.h>
#include <sys/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


void reduce_cpu_baseline(const float *input ,float *output,size_t n){
	float ret=0;
	for(int i = 0;i<n;i++){
		ret += input[i];
	}
	*output = ret;
}

__global__ void reduce_baseline(const float *input, float *output,size_t n){
	float sum = 0;
	for(int i = 0;i<n;i++){
		sum += input[i];
	}
	*output = sum;
}


int main(void){
	return 0;
}
