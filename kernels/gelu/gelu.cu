#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
struct GeluFunctor{
	static constexpr T alpha = static_cast<T>(0.7978845608028654);
	static constexpr T beta = static_cast<T>(0.7978845608028654);
	__device__ T operator()(T x) const{
		const T half = static_cast<T>(0.5);
		const T one = static_cast<T>(1);
		const T tanh_in = alpha*(x+beta*x*x*x);
		return half*x*(one + tanh(tanh_in));
		
	}
};

__global__ void gelu_baseline(const float *input,float *output,int n){
	int gtid = blockIdx.x*blockDim.x+threadIdx.x;
	GeluFunctor<float> gelu_fwd;
	if(gtid < n){
		output[gtid] = gelu_fwd(input[gtid]);
	}
}

int main(void){

}
