#include <vector>
#include <armadillo>
#include "vat/CUDA/MatrixUtil.cuh"
#include "Util.h"
#include <lapacke.h>

namespace vat {

template<typename T>
class AugmentedMatrix {
public:
	/*
	 * - Represents a linear system of the form:
	 * 	   Ax = b; A in T^nxn, b in T^nx1, and x is our solution vector.
	 *
	 * - T is a field of type:
	 *	   float, double, complex<float>, or complex<double>.
	 *
	 * */
	AugmentedMatrix(Mat<T>* _A, T* _b);

	~AugmentedMatrix();

	/*
	 * Summary: solve uses Gaussian Elimination (GE) to decompose and solve the system.
	 * Since b is unchanging, we use GE rather than LU Decomposition or
	 * Cholesky Factorization.
	 *
	 * Implementation details: Since both A and b are row-based matrices, we solve each
	 * column row-by-row.
	 * */
	T* solve(bool useGPU = true);

	//T* approxSolve();

private:
	Mat<T>* A;
	T* b;
	ul nRows, nCols;

	DiskCache* cache;
	BlockDims blockDims;
	GridDims gridDims;
	bool solvable;

	float* lapackLUSolve(float* AMat);
	double* lapackLUSolve(double* AMat);
	std::complex<float>* lapackLUSolve(std::complex<float>* AMat);
	std::complex<double>* lapackLUSolve(std::complex<double>* AMat);


	T* calculateRatios(T* rows, const ul col, const T inverse, const ul numRatios);

	void normalizeData(T* data, const ul normIdx = 0);

	void decomposeRows(T* rows, T* ratios, T* pivotRow, T* bView, const ul col, const ul numRows);

	//void decomposeCol(T* refRow, T* batch, T* B, ul pivot);s
	void decomposeCol(T* refRow, T* batch, T* B, T* ratios, ul pivot);

	const std::pair<T, vat::ul> argmax(T* data, const ul rowStart, const ul rowEnd, const ul col);

	std::tuple<bool, bool> getTypeTraits();

	T* rowMajorToColumnMajor(T* d, const int dim);

	/*bool hostSolveKey(T** keyRows, ul nKeyRowsTotal, ul nKeyCols, T* targetB);
	void hostSolveBlock(T** keyRows, ul nKeyRows, ul nKeyCols, T* keyB, T** targetRows, ul nTargetRows, T* targetB);
*/

	/*
	 * Summary: partialPivot is used when diag(i) = 0 where 0 < i < n_rows. It swaps an entire
	 * row i with another loaded row k such that data(i, i) << data(k, i). The swapped row is
	 * selected via PivotType.
	 *
	 * Heuristics: Given a 1e6 x 1e6 system of equations of type complex<double>,
	 * only 7500 (at max) rows can be read at one time on a 60GB system. Since the matrix is
	 * generally banded, it's recommended that you pick a PivotType that searches the upper portion
	 * of the matrix. Otherwise, partialPivot may not find a compatible row. Furthermore,
	 * FullSearches should not be used if the system is very large.
	 * */
	void partialPivot(T* refData, T* data, T* B, const ul zeroPivot);
	void partialPivot(T* refData, T* data, T* B, const ul zeroPivot, const ul iterations);

	bool isZero(T val);
	bool isOne(T val);

	float mag(float val);
	double mag(double val);
	float mag(std::complex<float> val);
	double mag(std::complex<double> val);

	T realScalar(double real);

	float multInverse(float a);
	double multInverse(double a);
	std::complex<float> multInverse(std::complex<float> a);
	std::complex<double> multInverse(std::complex<double> a);
};

//template class AugmentedMatrix<float>;
//template class AugmentedMatrix<double>;
template class AugmentedMatrix<std::complex<float>>;
template class AugmentedMatrix<std::complex<double>>;

template<> float* AugmentedMatrix<float>::lapackLUSolve(float* AMat);
template<> double* AugmentedMatrix<double>::lapackLUSolve(double* AMat);
template<> std::complex<float>* AugmentedMatrix<std::complex<float>>::lapackLUSolve(std::complex<float>* AMat);
template<> std::complex<double>* AugmentedMatrix<std::complex<double>>::lapackLUSolve(std::complex<double>* AMat);

template<> bool AugmentedMatrix<float>::isZero(float val);
template<> bool AugmentedMatrix<double>::isZero(double val);
template<> bool AugmentedMatrix<std::complex<float>>::isZero(std::complex<float> val);
template<> bool AugmentedMatrix<std::complex<double>>::isZero(std::complex<double> val);

template<> bool AugmentedMatrix<float>::isOne(float val);
template<> bool AugmentedMatrix<double>::isOne(double val);
template<> bool AugmentedMatrix<std::complex<float>>::isOne(std::complex<float> val);
template<> bool AugmentedMatrix<std::complex<double>>::isOne(std::complex<double> val);

template<> float AugmentedMatrix<float>::mag(float val);
template<> double AugmentedMatrix<double>::mag(double val);
template<> float AugmentedMatrix<std::complex<float>>::mag(std::complex<float> val);
template<> double AugmentedMatrix<std::complex<double>>::mag(std::complex<double> val);


template<> float AugmentedMatrix<float>::realScalar(double real);
template<> double AugmentedMatrix<double>::realScalar(double real);
template<> std::complex<float> AugmentedMatrix<std::complex<float>>::realScalar(double real);
template<> std::complex<double> AugmentedMatrix<std::complex<double>>::realScalar(double real);
}
