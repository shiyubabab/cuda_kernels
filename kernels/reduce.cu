#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

bool check_result(int *out,int groudtruth){
	if(*out != groudtruth) return false;
	return true;
}

__global__ void reduce_baseline(const int* input, int *output,size_t n){
	int sum = 0;
	for(size_t i = 0;i<n;++i) sum+= input[i];
	*output = sum;
}

int main(){

}
