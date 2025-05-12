#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define WarpSize 32

bool CheckResult(float *out,float *groudtruth,int N){
	for(int i = 0;i<N;i++){
		if(i == 0){
			printf("1st comparsion: %f and %f \n",out[i],groudtruth[i]);
		}
		if(out[i] != groudtruth[i]){
			return false;
		}
	}
	return true;
}

void softmax_cpu_baseline(float *input,float *result,int rows,int cols){
	for(int j = 0;j<rows;j++){
		float total = 0;
		float MAX = 0;
		for(int i = 0;i<cols;i++){
			MAX = max(input[j*cols+i],MAX);
		}
		for(int i = 0;i<cols;i++){
			total += exp(input[j*cols+i] - MAX);
		}
		for(int i = 0;i<cols;i++){
			result[j*cols+i] = exp(input[j*cols+i]-MAX)/total;
		}
	}
}



int mainl(void){

	return 0;
}
