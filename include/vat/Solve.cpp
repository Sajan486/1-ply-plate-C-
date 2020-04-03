#include "Solve.h"

template<typename T>
vat::AugmentedMatrix<T>::AugmentedMatrix(Mat<T>* _A, T* _b) {
	A = _A;
	b = _b;
	nRows = A->getN_Rows();
	nCols = A->getN_Cols();

	cache = A->getCache();
	blockDims = A->getBlockDims();
	gridDims  = A->getGridDims();

	solvable = true;
}

template<typename T>
vat::AugmentedMatrix<T>::~AugmentedMatrix() {}

template<typename T>
std::tuple<bool, bool> vat::AugmentedMatrix<T>::getTypeTraits() {
	bool cmplx = false;
	bool singlePrecision = false;
	ul typeSize = sizeof(T);
	if (std::is_same<T, std::complex<float>>::value  ||
	    std::is_same<T, std::complex<double>>::value) {
		cmplx = true;
		typeSize /= 2;
	}

	if (typeSize == sizeof(float)) {
		singlePrecision = true;
	}
	return std::make_tuple(singlePrecision, cmplx);
}

template<typename T>
T* vat::AugmentedMatrix<T>::rowMajorToColumnMajor(T* d, const int dim) {
	T* r = new T[dim * dim];
	for (int row = 0; row < dim; row++) {
		for (int col = 0; col < dim; col++) {
			r[row + col * dim] = d[col + row * dim];
		}
	}
	return r;
}

template<>
float* vat::AugmentedMatrix<float>::lapackLUSolve(float* AMat) {
	lapack_int n, nrhs, lda, ldb, info;

	n = nRows;
	nrhs = 1;
	lda = n;
	ldb = nrhs;
	lapack_int *ipiv;

	ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;

	info = LAPACKE_sgesv( LAPACK_ROW_MAJOR, n, nrhs, AMat, lda, ipiv, b, ldb );
	if (info > 0) {
		printf( "The diagonal element of the triangular factor of A,\n" );
	    printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
	    printf( "the solution could not be computed.\n" );
	    exit(1);
	}
	if (info < 0) {
		std::cout << "Info = " + std::to_string(info) + ". Illegal argument error." << std::endl;
		exit(1);
	} 
	return b;
}

template<>
double* vat::AugmentedMatrix<double>::lapackLUSolve(double* AMat) {
	lapack_int n, nrhs, lda, ldb, info;

	n = nRows;
	nrhs = 1;
	lda = n;
	ldb = nrhs;
	lapack_int *ipiv;

	ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;

	info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, AMat, lda, ipiv, b, ldb );
	if (info > 0) {
			printf( "The diagonal element of the triangular factor of A,\n" );
		    printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
		    printf( "the solution could not be computed.\n" );
		    exit(1);
		}
	if (info < 0) {
		std::cout << "Info = " + std::to_string(info) + ". Illegal argument error." << std::endl;
		exit(1);
	} 
	return b;
}

template<>
std::complex<float>* vat::AugmentedMatrix<std::complex<float>>::lapackLUSolve(std::complex<float>* AMat) {
	lapack_int n, nrhs, lda, ldb, info;

	n = nRows;
	nrhs = 1;
	lda = n;
	ldb = nrhs;
	lapack_int *ipiv;

	ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;

	info = LAPACKE_cgesv ( LAPACK_ROW_MAJOR, n, nrhs, (lapack_complex_float*)AMat, lda, ipiv, (lapack_complex_float*)b, ldb );

	if (info > 0) {
			printf( "The diagonal element of the triangular factor of A,\n" );
		    printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
		    printf( "the solution could not be computed.\n" );
		    exit(1);
	}
	if (info < 0) {
		std::cout << "Info = " + std::to_string(info) + ". Illegal argument error." << std::endl;
		exit(1);
	} 
	return b;
}

template<>
std::complex<double>* vat::AugmentedMatrix<std::complex<double>>::lapackLUSolve(std::complex<double>* AMat) {
	lapack_int n, nrhs, lda, ldb, info;

	n = nRows;
	nrhs = 1;
	lda = n;
	ldb = nrhs;
	lapack_int *ipiv;

	ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;

	info = LAPACKE_zgesv ( LAPACK_ROW_MAJOR, n, nrhs, (lapack_complex_double*)AMat, lda, ipiv, (lapack_complex_double*)b, ldb );
	if (info > 0) {
			printf( "The diagonal element of the triangular factor of A,\n" );
		    printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
		    printf( "the solution could not be computed.\n" );
		    exit(1);
		}
	if (info < 0) {
		std::cout << "Info = " + std::to_string(info) + ". Illegal argument error." << std::endl;
		exit(1);
	} 
	return b;
}


template<typename T>
T* vat::AugmentedMatrix<T>::solve(bool useGPU) {
	const ul maxElementsPerRead = cache->getMaxIO().nBytes() / sizeof(T) / 2;
	const ul maxRowsPerRead = maxElementsPerRead / nCols;
	ul rowsPerRead = maxElementsPerRead;
	//std::cout << sizeof(b) / sizeof(*b) << std::endl;

	//if ((maxElementsPerRead / 2) / nCols >= nRows) {
		if (useGPU) {
			T* data = A->subMat(0, 0, nRows - 1, nCols - 1);
			T* colA = rowMajorToColumnMajor(data, nRows);
			T* X = new T[nRows];

			std::tuple<bool, bool> typeTraits = getTypeTraits();
			bool singlePrecision = std::get<0>(typeTraits);
			bool cmplx           = std::get<1>(typeTraits);

			if (singlePrecision) {
				if (cmplx) {
					cuComplex* colAGPU = reinterpret_cast<cuComplex*> (colA);
					cuComplex* bGPU = reinterpret_cast<cuComplex*> (b);
					cuComplex* XGPU = reinterpret_cast<cuComplex*> (X);

					memLUDecomposition<cuComplex>(colAGPU, bGPU, XGPU, nRows);
				}
				else memLUDecomposition<float>((float *)colA, (float *)b, (float *)X, nRows);
			}
			else {
				if (cmplx) {
					cuDoubleComplex* colAGPU = reinterpret_cast<cuDoubleComplex*> (colA);
					cuDoubleComplex* bGPU = reinterpret_cast<cuDoubleComplex*> (b);
					cuDoubleComplex* XGPU = reinterpret_cast<cuDoubleComplex*> (X);
					memLUDecomposition<cuDoubleComplex>(colAGPU, bGPU, XGPU, nRows);
					}
				else memLUDecomposition<double>((double *)colA, (double*)b, (double*)X, nRows);
			}
			delete[] data;
			delete[] colA;
			return X;
		}
		else {
			std::cout << "Using lapacke solver" << std::endl;
			//Solve using Lapacke LU Decomposition
			T* subMat = A->subMat(0, 0, nRows - 1, nCols - 1);
			T* b = lapackLUSolve(subMat);
			delete subMat;
			return b;
		}
}