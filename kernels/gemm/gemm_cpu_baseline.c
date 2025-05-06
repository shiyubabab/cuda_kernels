/*************************************************************************
	> File Name: gemm_baseline.c
	> Author: mlxh
	> Mail: mlxh_gto@163.com 
	> Created Time: Tue 06 May 2025 01:41:11 PM CST
 ************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>

typedef float data_t;

typedef struct{
	int X;
	int Y;
	data_t **data;
} matrix_t ;

matrix_t matrix_init(const int _Y,const int _X){
	matrix_t ret;
	ret.X = _X;
	ret.Y = _Y;
	ret.data = (data_t **)malloc( _Y * sizeof(data_t *) );
	for(int i = 0;i<_Y;i++){
		ret.data[i] = (data_t *)malloc( _X * sizeof(data_t) );
	}
	return ret;
}

void matrix_dump(matrix_t *m){
	int _X = m->X;
	int _Y = m->Y;
	data_t **data = m->data;
	for(int y = 0; y<_Y; y++){
		for(int x = 0; x<_X; x++){
			printf("[%f]",data[y][x]);
		}
		printf("\n");
	}
}

void matrix_star(matrix_t *m,data_t number){
	int X = m->X;
	int Y = m->Y;
	for(int i = 0;i<Y;++i){
		for(int j = 0 ;j<X;++j){
			m->data[i][j] = number;
		}
	}
}


void matrix_gemm(matrix_t *A,matrix_t *B,matrix_t *C){
	assert(A->X == B->Y);
	assert(A->Y == C->Y);
	assert(B->X == C->X);
	int X = A -> Y;
	int Y = B -> X;
	int K = A -> X;
	data_t **Adata = A->data;
	data_t **Bdata = B->data;
	for(int y = 0;y<Y;y++){
		for(int x = 0;x<X;x++){
			int k = 0;
			data_t Cdata = 0 , Odata = 0;
			for(k;k<K;k+=2){
				Cdata += Adata[y][k] * Bdata[k][x];
				Odata += Adata[y][k+1] * Bdata[k+1][x];
			}
			for(;k<K;k++){
				Cdata += Adata[y][k] * Bdata[k][x];
			}
			C->data[y][x] = Cdata+Odata;
		}
	}
}


int main(void){
	matrix_t A = matrix_init(3,2);
	matrix_t B = matrix_init(2,3);
	matrix_t C = matrix_init(3,3);
	matrix_star(&A,3.14);
	matrix_star(&B,3);
	matrix_gemm(&A,&B,&C);
	matrix_dump(&C);
	return 0;
}

