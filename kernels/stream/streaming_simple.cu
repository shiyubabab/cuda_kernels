

int main(){
	cudaStream_t stream[3];
	float *A;
	float *d_A;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	for(int i = 0;i<3;i++) cudaStreamCreate(&stream[i]);
	cudaMalloc((void **)&d_A,30*sizeof(float));
	for(int i = 0;i<3;i++){
		cudaMemcpyAsync(d_A + i*10*sizeof(float),A+i*10*sizeof(float),
						10*sizeof(float),cudaMemcpyHostToDevice,stream[i]);

		cudaEventRecord(start);
		float_add << <10,1,0,stream[i]> >>(d_A + i*10*sizeof(float));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds,start,stop);

		cudaMemcpyAsync(d_A + i*10*sizeof(float),A+i*10*sizeof(float),
						10*sizeof(float),cudaMemcpyDeviceToHost,stream[i]);
	}
	for(int i = 0;i<3;i++) cudaStreamDestroy(stream[i]);
	cudaFreeHost(A);
	cudaFree(A);
}
