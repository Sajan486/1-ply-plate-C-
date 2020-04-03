#include "MatrixUtil.cuh"

typedef std::complex<float>  cmplxF;
typedef std::complex<double> cmplxD;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

cusolverStatus_t cusolverget_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      float *A,
                      int lda,
                      int *Lwork ) {
	return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverget_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      double *A,
                      int lda,
                      int *Lwork ) {
	return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverget_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      cuComplex *A,
                      int lda,
                      int *Lwork ) {
	return cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverget_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      cuDoubleComplex *A,
                      int lda,
                      int *Lwork ) {
	return cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolvergetrf(cusolverDnHandle_t handle,
           int m,
           int n,
           float *A,
           int lda,
           float *Workspace,
           int *devIpiv,
           int *devInfo) {
    return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t cusolvergetrf(cusolverDnHandle_t handle,
           int m,
           int n,
           double *A,
           int lda,
           double *Workspace,
           int *devIpiv,
           int *devInfo) {
    return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t cusolvergetrf(cusolverDnHandle_t handle,
           int m,
           int n,
           cuComplex *A,
           int lda,
           cuComplex *Workspace,
           int *devIpiv,
           int *devInfo) {
    return cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t cusolvergetrf(cusolverDnHandle_t handle,
           int m,
           int n,
           cuDoubleComplex *A,
           int lda,
           cuDoubleComplex *Workspace,
           int *devIpiv,
           int *devInfo) {
    return cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

cusolverStatus_t
cusolvergetrs(cusolverDnHandle_t handle,
           cublasOperation_t trans,
           int n,
           int nrhs,
           const float *A,
           int lda,
           const int *devIpiv,
           float *B,
           int ldb,
           int *devInfo ) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t
cusolvergetrs(cusolverDnHandle_t handle,
           cublasOperation_t trans,
           int n,
           int nrhs,
           const double *A,
           int lda,
           const int *devIpiv,
           double *B,
           int ldb,
           int *devInfo ) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t
cusolvergetrs(cusolverDnHandle_t handle,
           cublasOperation_t trans,
           int n,
           int nrhs,
           const cuComplex *A,
           int lda,
           const int *devIpiv,
           cuComplex *B,
           int ldb,
           int *devInfo ) {
    return cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

cusolverStatus_t
cusolvergetrs(cusolverDnHandle_t handle,
           cublasOperation_t trans,
           int n,
           int nrhs,
           const cuDoubleComplex *A,
           int lda,
           const int *devIpiv,
           cuDoubleComplex *B,
           int ldb,
           int *devInfo ) {
    return cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

template<typename T>
void vat::memLUDecomposition(T* rows, T* b, T* x, const int dim) {
	cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = dim;
    const int lda = dim;
    const int ldb = dim;

    T LU[lda*m]; /* L and U */

    int Ipiv[m];      /* host copy of pivoting sequence */
    int info = 0;     /* host copy of error info */

    T *d_A = NULL; /* device copy of A */
    T *d_B = NULL; /* device copy of B */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    T *d_work = NULL; /* device workspace for getrf */

    const int pivot_on = 1;

    printf("example of getrf \n");

    if (pivot_on){
        printf("pivot is on : compute P*A = L*U \n");
    }else{
        printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }

    printf("A = (matlab base-1)\n");
    //printMatrix(m, m, A, lda, "A");
    printf("=====\n");

    printf("B = (matlab base-1)\n");
   // printMatrix(m, 1, B, ldb, "B");
    printf("=====\n");

/* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(T) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(T) * m);
    cudaStat2 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, rows, sizeof(T)*lda*m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, b, sizeof(T)*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query working space of getrf */
    status = cusolverget_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(T)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 4: LU factorization */
    if (pivot_on){
        status = cusolvergetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            d_Ipiv,
            d_info);
    }else{
        status = cusolvergetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            NULL,
            d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    if (pivot_on){
    cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
    }
    cudaStat2 = cudaMemcpy(LU   , d_A   , sizeof(T)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on){
        printf("pivoting sequence, matlab base-1\n");
        for(int j = 0 ; j < m ; j++){
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
        }
    }
    //printf("L and U = (matlab base-1)\n");
    //printMatrix(m, m, LU, lda, "LU");
    //printf("=====\n");

    /*
 * step 5: solve A*X = B
 *       | 1 |       | -0.3333 |
 *   B = | 2 |,  X = |  0.6667 |
 *       | 3 |       |  0      |
 *
 */
    if (pivot_on){
        status = cusolvergetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_B,
            ldb,
            d_info);
    }else{
        status = cusolvergetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            NULL,
            d_B,
            ldb,
            d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(x , d_B, sizeof(T)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    //printf("X = (matlab base-1)\n");
    //printMatrix(m, 1, X, ldb, "X");
    //printf("=====\n");

    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_Ipiv) cudaFree(d_Ipiv);
    if (d_info) cudaFree(d_info);
    if (d_work) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);

    cudaDeviceReset();
}
template void vat::memLUDecomposition<float>(float* rows, float* b, float* x, const int dim);
template void vat::memLUDecomposition<double>(double* rows, double* b, double* x, const int dim);
template void vat::memLUDecomposition<cuComplex>(cuComplex* rows, cuComplex* b, cuComplex* x, const int dim);
template void vat::memLUDecomposition<cuDoubleComplex>(cuDoubleComplex* rows, cuDoubleComplex* b, cuDoubleComplex* x, const int dim);
