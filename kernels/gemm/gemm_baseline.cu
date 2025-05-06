



__global__ void block_mul(float *d_A, float *d_B, float *d_C){
	__shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	int tx = threadIdx.x,ty threadIdx.y;
	float P = 0;
	for(int i = 0;i<N/BLOCK_SIZE;i++){
		Mds[tx][ty] = d_A[row * N + ty + i * BLOCK_SIZE];
		Nds[tx][ty] = d_B[col + (tx + i*BLOCK_SIZE) * K];
		__syncthreads();
		for(int j = 0;j<BLOCK_SIZE;j++){
			P+=Mds[tx][j] * Nds[j][ty];
			__syncthreads();
		}

	}
	d_C[row * K + col] = P;
}
