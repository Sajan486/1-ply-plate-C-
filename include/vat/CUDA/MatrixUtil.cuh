#include <cuda_runtime.h>
#include "../include/Matrix.h"
#include <type_traits>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
#include <assert.h>

#define MAX_BLOCK_DIM 512

namespace vat {

template<typename T>
void memLUDecomposition(T* rows, T* b, T* x, const int dim);
}
