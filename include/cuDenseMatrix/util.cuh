#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <vector>
#include <ctime>

enum Inequality {
	LessThan = 0,
	LessThanOrEqual = 1,
	EqualTo = 2,
	GreaterThan = 3,
	GreaterThanOrEqual = 4,
	NotEqualTo = 5
};

enum CxOperation {
	Multiply = 0,
	Sum = 1,
	Power = 2,
	RealSqrt = 3,
	Exp = 4
};

/* Allocates arrays of any basic type to or from the host
 *
 * @param h_a:    Pointer to host array memory
 * @param d_a:    Pointer to device array memory
 * @param length: Number of elements in the array being copied
 * @param kind:   Specifies if memory will be copied to or from the host
 */
template<typename T>
void alloCopyArray(T*& h_a, T*& d_a, long length, cudaMemcpyKind kind);


/* Takes a single value of any type and allocates a pointer to that value on the device
 *
 * @param h_v:  Pointer to host memory
 * @param d_v:  Pointer to device memory
 * @param kind: Specifies if memory will be copied to or from the host
 */
template<typename T>
void alloCopyValue(T& hostMem, T*& deviceMem, cudaMemcpyKind kind);

template<typename F, typename uInt>
F* tiledOperationCxMat(F*& m, uInt nRows, uInt nCols, F*& t, uInt nRowsT, uInt nColsT, CxOperation op);

template<typename F, typename T, typename uInt>
F* elementOpCxMat(F*& m, uInt nRows, uInt nCols, T*& val, CxOperation op);

template<typename F>
F* inequalityCxMat(F*& m, long nRows, long nCols, F realVal, Inequality eq);

template<typename F, typename uInt>
F* cardExpansionCxMat(F*& m, uInt nRows, uInt nCols, uInt cardSize, bool horizontalExpand);

template<typename F>
F* reshapeCxMat(F*& m, long nRows, long nCols, long nRowsR, long nColsR, bool readHoriz, bool writeHoriz);

template<typename F>
F* submatCxMat(F*& m, long nCols, long r0, long c0, long rf, long cf);

template<typename F>
F* deckExpansionCxMat(F*& m, long nRows, long nCols, long cardSize, bool horizontalExpand);

template<typename F>
F* joinCxMat(F*& m, long nRows, long nCols, F*& s, long nRowSub, long nColSub, bool horizJoin);

template<typename F>
F* fillCxMat(F*& m, long nRows, long nCols, F realVal, F imagVal);

template<typename F, typename uInt>
F* dimensionReductionCxMat(F*& m, uInt nRows, uInt nCols, bool reduceRows);

template<typename F>
F* transposeCxMat(F*& m, long nRows, long nCols);

template<typename F, typename uInt>
void setByTiledPredicateCxMat(F*& pred, uInt nRowsP, uInt nColsP, Inequality eq, F testVal, F setVal, std::vector<F*>&, uInt nRows, uInt nCols);

template<typename F>
F* tiledDiagonalCxMat(F*& m, long nRows, long nCols, long tileSize, bool horizontalCompress);

template<typename F>
bool globalPredicateCxMat(F*& m, long nRows, long nCols, Inequality eq, F val);


/* -- Host side functions -- */


/* Returns the grid and block dims for a matrix so that each element has a thread
 *
 * @param nRows: Number of rows of the target matrix
 * @param nCols: Number of columns of the target matrix
 * @return:      A vector of size two with gridDims at index 0 and blockDims at index 1
 */
std::vector<dim3> matrixKernelDims(long nRows, long nCols);


/* Prints out a complex matrix
 *
 * @param m:      Dynamic array of complex floating points to print
 * @param nRows:  Number of rows of m
 * @param nCols:  Number of columns of m
 */
template<typename F>
void printCxMat(F*& m, long nRows, long nCols);

void profileCuFunctions(double duration);

void queryDeviceInfo();

void freeGlobalMemory();

void resetCudaTimes();

void checkMemoryUsage();

double getMaxMemoryUsage();

long getCoreCount(int deviceNumber);
