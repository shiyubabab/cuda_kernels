# cublas API record
## cublas 实现矩阵乘法
* cublas API也提供了一些帮助函数来写或者读取数据从GPU中
* 列优先的数组，索引以1为基准
* 头文件 include "cublas_v2.h"
* 三类函数（向量标量、向量举证、矩阵矩阵）
## 使用流程
* 准备A，B，C以及使用的线程网格、线程块的尺寸
* 创建句柄：cublasHandle_t handle; cublasCreate(&handle)；
* 调用计算函数：cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,*B,n,*A,k,&beta,*C,n);
* 销毁句柄：cublasDestroy（handle）；
* 回收计算结果
## level1函数：标量
### reduce函数：
* cublasStatus_t cublasamax(cublasHandle_t handle,int n,const float *x,int incx,int *result);
* cublasStatus_t cublasamin(cublasHandle_t handle,int n,const float *x,int incx,int *result);
## level2函数：矩阵向量
* cublasStatus_t cublasSgemv(cublasHandle_t handle,cublasOperation_t trans,int m,int n,const float *alpha,const float *A,int Ida,const float *x,int incx,const float *beta,float *y,int incy);
## level3函数：矩阵矩阵
* cublasStatus_t cublasSgemm(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const float *alpha,const float *A,int Ida,const float *B,int Idb,const float *beta,float *C,int Idc);

##cudnn 实现卷积神经网络
### cudnn是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。
### 常用的神经网络组件
* 前后向卷积网络
* 前后向pooling
* 前后向softmax
* 前后向神经元激活
* ReLU、TanH
* Tensor transformation functions
### 头文件 include "cudnn.h"
### 使用流程
* 创建cuDNN句柄：cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
* 以Host方式调用在Device上运行的函数: 比如卷积运算 cudnnConvolutionForward等
* 释放cuDNN句柄：cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
* 将CUDA流设置&返回成cudnn句柄
	* cudnnStatus_t cudnnSetStream(cudnnHandle_T handle,cudaStream_t streamld);
	* cudnnStatus_t cudnnGetStream(cudnnHandle_T handle,cudaStream_t streamld);


