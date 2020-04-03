#include "util.cuh"//

#define MAX_BLOCK_DIM 1024
#define MAX_BLOCK_SIZE 32

double tiledOpTime = 0;
double elementOpTime = 0;
double reductionTime = 0;
double inequalityTime = 0;
double cardExpansionTime = 0;
double reshapeTime = 0;
double submatTime = 0;
double deckExpansionTime = 0;
double joinTime = 0;
double fillTime = 0;
double alloCopyTime = 0;
double transposeTime = 0;
double tiledDiagonalTime = 0;
double globalPredicateTime = 0;

long maxMemoryUsage = 0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern __shared__ double sdata[];

template<typename F>
__device__ void complexProduct(F* cmplx, F r1, F i1, F r2, F i2) {

	F real = (r1 * r2) - (i1 * i2);
	F imag = (r1 * i2) + (r2 * i1);

	cmplx[0] = real;
	cmplx[1] = imag;
}

template<typename F>
__device__ void complexInverse(F* cmplx, F r, F i) {
	F real = cmplx[0];
	F imag = cmplx[1];

	F denom = real * real + imag * imag;
	if (denom != 0) {
		cmplx[0] = real / denom;
		cmplx[1] = -imag / denom;
	}
}

template<typename F, typename T>
__device__ void complexPower(F* cmplx, F r1, F i1, T exponent) {

	F origReal = cmplx[0];
	F origImag = cmplx[1];

	bool inverse = (exponent < 0);
	exponent = abs(exponent);

	for (T i = 0; i < (exponent - 1); i++) {
		complexProduct<F>(cmplx, cmplx[0], cmplx[1], origReal, origImag);
	}

	if (inverse) {
		complexInverse<F>(cmplx, cmplx[0], cmplx[1]);
	}
}

template<typename F>
__device__ void complexExp(F* cmplx, F mReal, F mImag) {

	F ex = exp(mReal);

	cmplx[0] = ex * cos(mImag);
	cmplx[1] = ex * sin(mImag);
}

template<typename F, typename T>
__device__ void complexSum(F* cmplx, F r1, F i1, T r2, T i2) {
	F real = r1 + r2;
	F imag = i1 + i2;

	cmplx[0] = real;
	cmplx[1] = imag;
}

template<typename F, typename uInt>
__global__ void cuTiledRowMulCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;
	if (mReal == 0 && mImag == 0) return;

	uInt yTile = yIdx % *nRowsT;
	uInt globalIdxT = xIdx + yTile * (*nColsT);

	idx = 2 * globalIdxT;
	F tReal = t[idx];
	F tImag = t[idx + 1];

	*mRealPtr = mReal * tReal - mImag * tImag;
	*mImagPtr = mReal * tImag + tReal * mImag;
}

template<typename F, typename uInt>
__global__ void cuTiledColMulCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;
	if (mReal == 0 && mImag == 0) return;

	uInt xTile = xIdx % *nColsT;
	uInt globalIdxT = xTile + yIdx * (*nColsT);

	idx = 2 * globalIdxT;
	F tReal = t[idx];
	F tImag = t[idx + 1];

	*mRealPtr = mReal * tReal - mImag * tImag;
	*mImagPtr = mReal * tImag + tReal * mImag;
}

template<typename F, typename uInt>
__global__ void cuRowMulCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;
	if (mReal == 0 && mImag == 0) return;

	idx = 2 * xIdx;
	F tReal = t[idx];
	F tImag = t[idx + 1];

	*mRealPtr = mReal * tReal - mImag * tImag;
	*mImagPtr = mReal * tImag + tReal * mImag;
}

template<typename F, typename uInt>
__global__ void cuColMulCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;
	if (mReal == 0 && mImag == 0) return;

	idx = 2 * yIdx;
	F tReal = t[idx];
	F tImag = t[idx + 1];

	*mRealPtr = mReal * tReal - mImag * tImag;
	*mImagPtr = mReal * tImag + tReal * mImag;
}

template<typename F, typename uInt>
__global__ void cuTiledRowSumCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt yTile = yIdx % *nRowsT;
	uInt globalIdxT = xIdx + yTile * (*nColsT);

	uInt idx = 2 * globalIdxT;
	F tReal = t[idx];
	F tImag = t[idx + 1];
	if (tReal == 0 && tImag == 0) return;

	idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;

	*mRealPtr = mReal + tReal;
	*mImagPtr = mImag + tImag;
}

template<typename F, typename uInt>
__global__ void cuTiledColSumCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt xTile = xIdx % *nColsT;
	uInt globalIdxT = xTile + yIdx * (*nColsT);

	uInt idx = 2 * globalIdxT;
	F tReal = t[idx];
	F tImag = t[idx + 1];
	if (tReal == 0 && tImag == 0) return;

	idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;

	*mRealPtr = mReal + tReal;
	*mImagPtr = mImag + tImag;
}

template<typename F, typename uInt>
__global__ void cuRowSumCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * xIdx;
	F tReal = t[idx];
	F tImag = t[idx + 1];
	if (tReal == 0 && tImag == 0) return;

	idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;

	*mRealPtr = mReal + tReal;
	*mImagPtr = mImag + tImag;
}

template<typename F, typename uInt>
__global__ void cuColSumCxMat(F* m, uInt* nRows, uInt* nCols, F* t, uInt* nRowsT, uInt* nColsT) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows - 1) return;

	uInt idx = 2 * yIdx;
	F tReal = t[idx];
	F tImag = t[idx + 1];
	if (tReal == 0 && tImag == 0) return;

	idx = 2 * globalIdx;
	F* mRealPtr = m + idx + 0;
	F* mImagPtr = mRealPtr + 1;
	F mReal = *mRealPtr;
	F mImag = *mImagPtr;

	*mRealPtr = mReal + tReal;
	*mImagPtr = mImag + tImag;
}

template<typename F, typename T, typename uInt>
__global__ void cuMultiplyCxMat(F* m, uInt* nRows, uInt* nCols, T* cxScalar) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = 2 * (xIdx + yIdx * (*nCols));

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	T mReal = m[globalIdx];
	T mImag = m[globalIdx + 1];

	T sReal = cxScalar[0];
	T sImag = cxScalar[1];

	m[globalIdx    ] = mReal * sReal - mImag * sImag;
	m[globalIdx + 1] = mReal * sImag + sReal * mImag;
}

template<typename F, typename T, typename uInt>
__global__ void cuPowerCxMat(F* m, uInt* nRows, uInt* nCols, T* exponent) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F mReal = m[2 * globalIdx + 0];
	F mImag = m[2 * globalIdx + 1];

	F cmplx[2];
	cmplx[0] = mReal;
	cmplx[1] = mImag;

	complexPower<F, T>(cmplx, mReal, mImag, *exponent);

	m[2 * globalIdx + 0] = cmplx[0];
	m[2 * globalIdx + 1] = cmplx[1];
}

template<typename F, typename uInt>
__global__ void cuRealSqrtCxMat(F* m, uInt* nRows, uInt* nCols) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	m[2 * globalIdx + 0] = sqrt(m[2 * globalIdx + 0]);
}

template<typename F, typename uInt>
__global__ void cuExpCxMat(F* m, uInt* nRows, uInt* nCols) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F mReal = m[2 * globalIdx + 0];
	F mImag = m[2 * globalIdx + 1];

	F cmplx[2];
	cmplx[0] = mReal;
	cmplx[1] = mImag;

	complexExp<F>(cmplx, mReal, mImag);

	m[2 * globalIdx + 0] = cmplx[0];
	m[2 * globalIdx + 1] = cmplx[1];
}

template<typename F, typename T, typename uInt>
__global__ void cuSumCxMat(F* m, uInt* nRows, uInt* nCols, T* cxScalar) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F mReal = m[2 * globalIdx + 0];
	F mImag = m[2 * globalIdx + 1];

	T sReal = cxScalar[0];
	T sImag = cxScalar[1];

	F cmplx[2];

	complexSum<F, T>(cmplx, mReal, mImag, sReal, sImag);

	m[2 * globalIdx + 0] = cmplx[0];
	m[2 * globalIdx + 1] = cmplx[1];
}

template<typename F>
__global__ void cuRealIsGreaterThanCxMat(F* m, long* nRows, long* nCols, F* realVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F currentRealVal = m[2 * globalIdx];
	if (*realVal < currentRealVal) m[2 * globalIdx + 0] = 1.0;
	else m[2 * globalIdx + 0] = 0.0;
	m[2 * globalIdx + 1] = 0.0;
}

template<typename F>
__global__ void cuRealIsGreaterThanOrEqualCxMat(F* m, long* nRows, long* nCols, F* realVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F currentRealVal = m[2 * globalIdx];
	if (*realVal <= currentRealVal) m[2 * globalIdx + 0] = 1.0;
	else m[2 * globalIdx + 0] = 0.0;
	m[2 * globalIdx + 1] = 0.0;
}

template<typename F>
__global__ void cuRealIsLessThanCxMat(F* m, long* nRows, long* nCols, F* realVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F currentRealVal = m[2 * globalIdx];
	if (*realVal > currentRealVal) m[2 * globalIdx + 0] = 1.0;
	else m[2 * globalIdx + 0] = 0.0;
	m[2 * globalIdx + 1] = 0.0;
}

template<typename F>
__global__ void cuRealIsLessThanOrEqualCxMat(F* m, long* nRows, long* nCols, F* realVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F currentRealVal = m[2 * globalIdx];
	if (*realVal >= currentRealVal) m[2 * globalIdx + 0] = 1.0;
	else m[2 * globalIdx + 0] = 0.0;
	m[2 * globalIdx + 1] = 0.0;
}

template<typename F>
__global__ void cuRealIsEqualToCxMat(F* m, long* nRows, long* nCols, F* realVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	F currentRealVal = m[2 * globalIdx];
	if (*realVal == currentRealVal) m[2 * globalIdx + 0] = 1.0;
	else m[2 * globalIdx + 0] = 0.0;
	m[2 * globalIdx + 1] = 0.0;
}

template<typename F, typename uInt>
__global__ void cuHorizontalCardExpansion(F* p, F* m, uInt* nCols, uInt* nRowsExp, uInt* nColsExp, uInt* cardSize) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIdx > *nColsExp - 1 || yIdx > *nRowsExp -  1) return;

	uInt gIdx = 2 * ((xIdx / *cardSize) + yIdx * (*nCols));
	uInt idx = 2 * (xIdx + yIdx * (*nColsExp));

	p[idx    ] = m[gIdx    ];
	p[idx + 1] = m[gIdx + 1];
}

template<typename F, typename uInt>
__global__ void cuVerticalCardExpansion(F* p, F* m, uInt* nRowsExp, uInt* nColsExp, uInt* cardSize) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIdx > *nColsExp - 1 || yIdx > *nRowsExp -  1) return;

	uInt gIdx = 2 * (xIdx + (yIdx / *cardSize) * (*nColsExp));
	uInt idx = 2 * (xIdx + yIdx * (*nColsExp));

	p[idx    ] = m[gIdx    ];
	p[idx + 1] = m[gIdx + 1];
}

template<typename F>
__global__ void cuReshapeCxMat(F* p, F* m, long* nRows, long* nCols, long* nRowsR, long* nColsR, bool* readHoriz, bool* writeHoriz) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	long readIdx;
	long xIdxRd;
	long yIdxRd;

	if (*readHoriz) {
		xIdxRd = xIdx;
		yIdxRd = yIdx;
	}
	else { // read vertical
		xIdxRd = globalIdx / (*nRows);
		yIdxRd = globalIdx - (xIdxRd * (*nRows));
	}

	readIdx = xIdxRd + yIdxRd * (*nCols);

	long writeIdx;
	long xIdxWr;
	long yIdxWr;

	if (*writeHoriz) {
		yIdxWr = globalIdx / (*nColsR);
		xIdxWr = globalIdx - (yIdxWr * (*nColsR));

	} else { // Write vertical
		xIdxWr = globalIdx / (*nRowsR);
		yIdxWr = globalIdx - (xIdxWr * (*nRowsR));
	}

	writeIdx = xIdxWr + yIdxWr * (*nColsR);

	p[2 * writeIdx + 0] = m[2 * readIdx + 0];
	p[2 * writeIdx + 1] = m[2 * readIdx + 1];
}

template<typename F>
__global__ void cuSubmatCxMat(F* p, F* m, long* nCols, long* nRowSub, long* nColSub, long* r0, long* c0) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nColSub);

	if (xIdx > *nColSub - 1 || yIdx > *nRowSub -  1) return;

	long xIdxM = xIdx + (*c0);
	long yIdxM = yIdx + (*r0);

	long idxM = xIdxM + yIdxM * (*nCols);

	p[2 * globalIdx + 0] = m[2 * idxM + 0];
	p[2 * globalIdx + 1] = m[2 * idxM + 1];
}

template<typename F>
__global__ void cuDeckExpansionCxMat(F* p, F* m, long* nRows, long* nCols, long* nColsExp, long* nRowsExp) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nColsExp);

	if (xIdx > *nColsExp - 1 || yIdx > *nRowsExp -  1) return;

	long xIdxM = xIdx - (xIdx / *nCols) * (*nCols);
	long yIdxM = yIdx - (yIdx / *nRows) * (*nRows);
	long globalIdxM = xIdxM + yIdxM * (*nCols);

	p[2 * globalIdx + 0] = m[2 * globalIdxM + 0];
	p[2 * globalIdx + 1] = m[2 * globalIdxM + 1];
}

template<typename F>
__global__ void cuJoinHorizontalCxMat(F* p, F* m, F* s, long* nRows, long* nCols, long* nColSub) {

	long nColsT = (*nCols) + (*nColSub);

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (nColsT);

	if (xIdx > nColsT - 1 || yIdx > *nRows -  1) return;

	long readIdx;

	if (xIdx < *nCols) {

		readIdx = xIdx + yIdx * (*nCols);

		p[2 * globalIdx + 0] = m[2 * readIdx + 0];
		p[2 * globalIdx + 1] = m[2 * readIdx + 1];
	} else {

		readIdx = (xIdx - (*nCols)) + yIdx * (*nColSub);

		p[2 * globalIdx + 0] = s[2 * readIdx + 0];
		p[2 * globalIdx + 1] = s[2 * readIdx + 1];
	}
}

template<typename F>
__global__ void cuJoinVerticalCxMat(F* p, F* m, F* s, long* nRows, long* nCols, long* nRowSub) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	long nRowsT = (*nRows) + (*nRowSub);

	if (xIdx > *nCols - 1 || yIdx > nRowsT -  1) return;

	if (yIdx < *nRows) {
		p[2 * globalIdx + 0] = m[2 * globalIdx + 0];
		p[2 * globalIdx + 1] = m[2 * globalIdx + 1];

	} else {

		long readIdx = xIdx + (yIdx - (*nRows)) * (*nCols);

		p[2 * globalIdx + 0] = s[2 * readIdx + 0];
		p[2 * globalIdx + 1] = s[2 * readIdx + 1];
	}
}

template<typename F>
__global__ void cuFillCxMat(F* m, long* nRows, long* nCols, F* realVal, F* imagVal) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	m[2 * globalIdx + 0] = *realVal;
	m[2 * globalIdx + 1] = *imagVal;
}

// switch all x's to y's and vis versa
template<typename F, typename uInt, unsigned int blockSize>
__global__ void rowReductionCxMat(F* p, F* m, uInt* nElements) {

	extern __shared__ double sdata[]; // TODO: implement for floats

	uInt tid = threadIdx.y;
	uInt xIdx = threadIdx.y + blockIdx.y * (2 * blockDim.y);
	uInt i = xIdx + blockIdx.x * (*nElements);

	sdata[tid] = 0;
	uInt gridSize = 2 * blockSize * gridDim.y;
	uInt endI = *nElements * (blockIdx.x + 1);

	while (i < endI) {
		if (i + blockDim.y < endI) {
			sdata[tid] += m[i] + m[i + blockSize];
		} else {
			sdata[tid] += m[i];
		}
		i += gridSize;
	} __syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		}
	} __syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
	} __syncthreads();

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
	} __syncthreads();

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
	} __syncthreads();

	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=  4) sdata[tid] += sdata[tid +  2]; // we skip the last reduction to preseve complex values
	}

	if (tid < 2) {
		p[2 * (blockIdx.y + gridDim.y * blockIdx.x) + tid] = sdata[tid];
	}
}

template<typename F, typename uInt, unsigned int blockSize>
__global__ void columnReductionCxMat(F* p, F* m, uInt* nElements) {

	extern __shared__ double sdata[];

	uInt tid = threadIdx.y;
	uInt yIdx = threadIdx.y + blockIdx.y * (2 * blockDim.y);
	uInt i = blockIdx.x + yIdx * gridDim.x;

	sdata[tid] = 0;
	uInt gridSize = 2 * blockSize * gridDim.x;
	uInt halfGridSize = blockSize * gridDim.x;
	uInt endI = gridDim.x * (*nElements);

	while (i < endI) {
		if (i + halfGridSize < endI) {
			sdata[tid] += m[i] + m[i + halfGridSize];
		} else {
			sdata[tid] += m[i];
		}
		i += gridSize;
	} __syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		}
	} __syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
	} __syncthreads();

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
	} __syncthreads();

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
	} __syncthreads();

	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
	}

	if (tid == 0) {
		p[blockIdx.x] = sdata[tid];
	}
}

template<typename F, int tilesPerBlock>
__global__ void cuTransposeCxMat(F* p, F* m, long* nRows, long* nCols) {

	long xIdx = threadIdx.x + tilesPerBlock * blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;

	for (int j = 0; j < tilesPerBlock; j++) {

		if (xIdx < *nCols && yIdx < *nRows) {

			long mIdx = xIdx + yIdx * (*nCols);
			long pIdx = yIdx + xIdx * (*nRows);

			p[2 * pIdx + 0] = m[2 * mIdx + 0];
			p[2 * pIdx + 1] = m[2 * mIdx + 1];

		}
		xIdx += blockDim.x;
	}
}

template<typename F, typename uInt, unsigned int nTargets>
__global__ void cuTiledIsGreaterThan(F* pred, uInt* nRowsP, uInt* nColsP, F* testVal, F* setVal, uInt* nRows, uInt* nCols, F** targetM) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt idx = 2 * (xIdx + yIdx * (*nCols));
	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	uInt xIdxP = xIdx % *nColsP;
	uInt yIdxP = yIdx % *nRowsP;
	uInt globalIdxP = xIdxP + yIdxP * (*nColsP);

	if (nTargets == 1) {
		if (pred[2 * globalIdxP] > (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
		}
	}

	if (nTargets == 2) {
		if (pred[2 * globalIdxP] > (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
		}
	}

	if (nTargets == 3) {
		if (pred[2 * globalIdxP] > (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
			targetM[2][idx    ] = *setVal;
			targetM[2][idx + 1] = *setVal;
		}
	}
}

template<typename F, typename uInt, unsigned int nTargets>
__global__ void cuTiledIsLessThan(F* pred, uInt* nRowsP, uInt* nColsP, F* testVal, F* setVal, uInt* nRows, uInt* nCols, F** targetM) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt idx = 2 * (xIdx + yIdx * (*nCols));
	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	uInt xIdxP = xIdx % *nColsP;
	uInt yIdxP = yIdx % *nRowsP;
	uInt globalIdxP = xIdxP + yIdxP * (*nColsP);

	if (nTargets == 1) {
		if (pred[2 * globalIdxP] < (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
		}
	}

	if (nTargets == 2) {
		if (pred[2 * globalIdxP] < (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
		}
	}

	if (nTargets == 3) {
		if (pred[2 * globalIdxP] < (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
			targetM[2][idx    ] = *setVal;
			targetM[2][idx + 1] = *setVal;
		}
	}
}

template<typename F, typename uInt, unsigned int nTargets>
__global__ void cuTiledIsNotEqualTo(F* pred, uInt* nRowsP, uInt* nColsP, F* testVal, F* setVal, uInt* nRows, uInt* nCols, F** targetM) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt idx = 2 * (xIdx + yIdx * (*nCols));
	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	uInt xIdxP = xIdx - (xIdx / *nColsP) * (*nColsP);
	uInt yIdxP = yIdx - (yIdx / *nRowsP) * (*nRowsP);
	uInt globalIdxP = xIdxP + yIdxP * (*nColsP); // TODO: simplify globalIdxs as with others

	if (nTargets == 1) {
		if (pred[2 * globalIdxP] != (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
		}
	}

	if (nTargets == 2) {
		if (pred[2 * globalIdxP] != (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
		}
	}

	if (nTargets == 3) {
		if (pred[2 * globalIdxP] != (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
			targetM[2][idx    ] = *setVal;
			targetM[2][idx + 1] = *setVal;
		}
	}
}

template<typename F, typename uInt, unsigned int nTargets>
__global__ void cuTiledIsEqualTo(F* pred, uInt* nRowsP, uInt* nColsP, F* testVal, F* setVal, uInt* nRows, uInt* nCols, F** targetM) {

	uInt xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	uInt yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	uInt idx = 2 * (xIdx + yIdx * (*nCols));
	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	uInt xIdxP = xIdx - (xIdx / *nColsP) * (*nColsP);
	uInt yIdxP = yIdx - (yIdx / *nRowsP) * (*nRowsP);
	uInt globalIdxP = xIdxP + yIdxP * (*nColsP); // TODO: simplify globalIdxs as with others

	if (nTargets == 1) {
		if (pred[2 * globalIdxP] == (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
		}
	}

	if (nTargets == 2) {
		if (pred[2 * globalIdxP] == (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
		}
	}

	if (nTargets == 3) {
		if (pred[2 * globalIdxP] == (*testVal)) {
			targetM[0][idx    ] = *setVal;
			targetM[0][idx + 1] = *setVal;
			targetM[1][idx    ] = *setVal;
			targetM[1][idx + 1] = *setVal;
			targetM[2][idx    ] = *setVal;
			targetM[2][idx + 1] = *setVal;
		}
	}
}

template<typename F>
__global__ void cuTiledDiagonalCxMat(F* p, F* m, long* nRows, long* nCols, long* xTileSize, long* yTileSize) {

	long xIdxP = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdxP = threadIdx.y + blockIdx.y * blockDim.y;
	long pIdx  = xIdxP + yIdxP * (*nCols / *xTileSize);

	long xTileM = yIdxP % *xTileSize;
	long yTileM = xIdxP % *yTileSize;
	long xIdxM  = xIdxP * (*xTileSize) + xTileM;
	long yIdxM  = yIdxP * (*yTileSize) + yTileM;

	if (xIdxM > *nCols - 1 || yIdxM > *nRows - 1) return;

	long mIdx = xIdxM + yIdxM * (*nCols);

	p[2 * pIdx + 0] = m[2 * mIdx + 0];
	p[2 * pIdx + 1] = m[2 * mIdx + 1];
}

template<typename F>
__global__ void cuGlobalEqualToCxMat(F* m, long* nRows, long* nCols, F* val, unsigned int* isTrue) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	long idx = 2 * globalIdx;

	F real = m[idx + 0];
	F imag = m[idx + 1];

	if (real != *val || imag != *val) {
		*isTrue = 0;
	}
}

template<typename F>
__global__ void cuGlobalNotEqualToCxMat(F* m, long* nRows, long* nCols, F* val, unsigned int* isTrue) {

	long xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	long yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	long globalIdx = xIdx + yIdx * (*nCols);

	if (xIdx > *nCols - 1 || yIdx > *nRows -  1) return;

	long idx = 2 * globalIdx;

	F real = m[idx + 0];
	F imag = m[idx + 1];

	if (real == *val && imag == *val) {
		*isTrue = 0;
	}
}

template<typename T>
void alloCopyArray(T*& h_a, T*& d_a, long length, cudaMemcpyKind kind) {

	std::clock_t start;
	start = std::clock();

	size_t size = length * sizeof(T);

	if (kind == cudaMemcpyHostToDevice) {
		gpuErrchk(cudaMalloc((void**) &d_a, size));
		gpuErrchk(cudaMemcpy(d_a, h_a, size, kind));
	}
	else if (kind == cudaMemcpyDeviceToHost) {
		h_a = (T*) malloc(size);
		gpuErrchk(cudaMemcpy(h_a, d_a, size, kind));
	}

	alloCopyTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;
}
template void alloCopyArray<double>(double*&, double*&, long, cudaMemcpyKind);
template void alloCopyArray<int>(int*&, int*&, long, cudaMemcpyKind);
template void alloCopyArray<long>(long*&, long*&, long, cudaMemcpyKind);
template void alloCopyArray<bool>(bool*&, bool*&, long, cudaMemcpyKind);

template<typename T>
void alloCopyValue(T& h_v, T*& d_v, cudaMemcpyKind kind) {

	if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost) {
		T* h_vA = new T[1];
		h_vA[0] = h_v;

		alloCopyArray(h_vA, d_v, 1, kind);
		h_v = h_vA[0];
		delete[] h_vA;
	}
}
template void alloCopyValue<float>(float&, float*&, cudaMemcpyKind);
template void alloCopyValue<double>(double&, double*&, cudaMemcpyKind);
template void alloCopyValue<bool>(bool&, bool*&, cudaMemcpyKind);
template void alloCopyValue<int>(int&, int*&, cudaMemcpyKind);
template void alloCopyValue<long>(long&, long*&, cudaMemcpyKind);
template void alloCopyValue<unsigned int>(unsigned int&, unsigned int*&, cudaMemcpyKind);


template<typename F, typename uInt>
F* tiledOperationCxMat(F*& m, uInt nRows, uInt nCols, F*& t, uInt nRowsT, uInt nColsT, CxOperation op) {

	std::clock_t start;
	start = std::clock();

	bool tileByRow;
	uInt tileWidth;

	if (nColsT == nCols) {
		tileByRow = true;
		tileWidth = nRowsT;
	} else {
		tileByRow = false;
		tileWidth = nColsT;
	}

	uInt* d_nRows;
	uInt* d_nCols;
	uInt* d_nRowsT;
	uInt* d_nColsT;

	alloCopyValue<uInt>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nRowsT, d_nRowsT, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nColsT, d_nColsT, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	if (op == Multiply) {
		if (tileByRow) {
			if (tileWidth > 1) {
				cuTiledRowMulCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			} else {
				cuRowMulCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			}
		} else {
			if (tileWidth > 1) {
				cuTiledColMulCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			} else {
				cuColMulCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			}
		}
	} else if (op == Sum) {
		if (tileByRow) {
			if (tileWidth > 1) {
				cuTiledRowSumCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			} else {
				cuRowSumCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			}
		} else {
			if (tileWidth > 1) {
				cuTiledColSumCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			} else {
				cuColSumCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, t, d_nRowsT, d_nColsT);
			}
		}
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowsT));
	gpuErrchk(cudaFree(d_nColsT));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	tiledOpTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return m;
}
template float* tiledOperationCxMat<float,unsigned long>(float*&, unsigned long, unsigned long, float*&, unsigned long, unsigned long, CxOperation);
template double* tiledOperationCxMat<double,unsigned long>(double*&, unsigned long, unsigned long, double*&, unsigned long, unsigned long, CxOperation);
template float* tiledOperationCxMat<float,unsigned int>(float*&, unsigned int, unsigned int, float*&, unsigned int, unsigned int, CxOperation);
template double* tiledOperationCxMat<double,unsigned int>(double*&, unsigned int, unsigned int, double*&, unsigned int, unsigned int, CxOperation);

template<typename F, typename T, typename uInt>
F* elementOpCxMat(F*& m, uInt nRows, uInt nCols, T*& val, CxOperation op) {

	std::clock_t start;
	start = std::clock();

	uInt* d_nRows;
	uInt* d_nCols;

	alloCopyValue<uInt>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nRows, d_nRows, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	if (op == Multiply) {
		T* d_cxScalar;
		alloCopyArray<T>(val, d_cxScalar, 2, cudaMemcpyHostToDevice);

		cuMultiplyCxMat<F,T,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_cxScalar);
	} else if (op == Power) {
		T* d_exponent;
		alloCopyValue<T>(*val, d_exponent, cudaMemcpyHostToDevice);
		cuPowerCxMat<F,T,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_exponent);
	} else if (op == RealSqrt) {
		cuRealSqrtCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols);
	} else if (op == Exp) {
		cuExpCxMat<F,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols);
	} else if (op == Sum) {
		T* d_cxScalar;
		alloCopyArray<T>(val, d_cxScalar, 2, cudaMemcpyHostToDevice);

		cuSumCxMat<F,T,uInt><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_cxScalar);
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	elementOpTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return m;
}
template float* elementOpCxMat<float,float,unsigned int>(float*&, unsigned int, unsigned int, float*&, CxOperation);
template double* elementOpCxMat<double,double,unsigned int>(double*&, unsigned int, unsigned int, double*&, CxOperation);
template float* elementOpCxMat<float,int,unsigned int>(float*&, unsigned int, unsigned int, int*&, CxOperation);
template double* elementOpCxMat<double,int,unsigned int>(double*&, unsigned int, unsigned int, int*&, CxOperation);
template float* elementOpCxMat<float,float,unsigned long>(float*&, unsigned long, unsigned long, float*&, CxOperation);
template double* elementOpCxMat<double,double,unsigned long>(double*&, unsigned long, unsigned long, double*&, CxOperation);
template float* elementOpCxMat<float,int,unsigned long>(float*&, unsigned long, unsigned long, int*&, CxOperation);
template double* elementOpCxMat<double,int,unsigned long>(double*&, unsigned long, unsigned long, int*&, CxOperation);

template<typename F>
F* inequalityCxMat(F*& m, long nRows, long nCols, F realVal, Inequality eq) {

	std::clock_t start;
	start = std::clock();

	long* d_nRows;
	long* d_nCols;
	F* d_realVal;
	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<F>(realVal, d_realVal, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	if (eq == GreaterThan) {
		cuRealIsGreaterThanCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal);
	} else if (eq == GreaterThanOrEqual) {
		cuRealIsGreaterThanOrEqualCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal);
	} else if (eq == LessThan) {
		cuRealIsLessThanCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal);
	} else if (eq == LessThanOrEqual){
		cuRealIsLessThanOrEqualCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal);
	} else if (eq == EqualTo){
		cuRealIsEqualToCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal);
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_realVal));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	inequalityTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return m;
}
template float* inequalityCxMat<float>(float*&, long, long, float, Inequality);
template double* inequalityCxMat<double>(double*&, long, long, double, Inequality);

template<typename F, typename uInt>
F* cardExpansionCxMat(F*& m, uInt nRows, uInt nCols, uInt cardSize, bool horizontalExpand) {

	std::clock_t start;
	start = std::clock();

	uInt nRowsExp = nRows;
	uInt nColsExp = nCols;

	if (horizontalExpand) nColsExp = nCols * cardSize;
	else nRowsExp = nRows * cardSize;

	uInt* d_nCols;
	uInt* d_nRowsExp;
	uInt* d_nColsExp;
	uInt* d_cardSize;
	bool* d_horizExp;
	alloCopyValue<uInt>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nRowsExp, d_nRowsExp, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nColsExp, d_nColsExp, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(cardSize, d_cardSize, cudaMemcpyHostToDevice);
	alloCopyValue<bool>(horizontalExpand, d_horizExp, cudaMemcpyHostToDevice);

	F* d_p;
	gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRowsExp * nColsExp * sizeof(F)));

	std::vector<dim3> kDims = matrixKernelDims(nRowsExp, nColsExp);

	if (horizontalExpand) {
		cuHorizontalCardExpansion<F,uInt><<<kDims[0], kDims[1]>>>(d_p, m, d_nCols, d_nRowsExp, d_nColsExp, d_cardSize);
	} else {
		cuVerticalCardExpansion<F,uInt><<<kDims[0], kDims[1]>>>(d_p, m, d_nRowsExp, d_nColsExp, d_cardSize);
	}

	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowsExp));
	gpuErrchk(cudaFree(d_nColsExp));
	gpuErrchk(cudaFree(d_cardSize));
	gpuErrchk(cudaFree(d_horizExp));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cardExpansionTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* cardExpansionCxMat<float,unsigned int>(float*&, unsigned int, unsigned int, unsigned int, bool);
template double* cardExpansionCxMat<double,unsigned int>(double*&, unsigned int, unsigned int, unsigned int, bool);
template float* cardExpansionCxMat<float,unsigned long>(float*&, unsigned long, unsigned long, unsigned long, bool);
template double* cardExpansionCxMat<double,unsigned long>(double*&, unsigned long, unsigned long, unsigned long, bool);

template<typename F>
F* reshapeCxMat(F*& m, long nRows, long nCols, long nRowsR, long nColsR, bool readHoriz, bool writeHoriz) {

	std::clock_t start;
	start = std::clock();

	F* d_p;

	long* d_nRows;
	long* d_nCols;
	long* d_nRowsR;
	long* d_nColsR;
	bool* d_readHoriz;
	bool* d_writeHoriz;

	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nRowsR, d_nRowsR, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nColsR, d_nColsR, cudaMemcpyHostToDevice);
	alloCopyValue<bool>(readHoriz, d_readHoriz, cudaMemcpyHostToDevice);
	alloCopyValue<bool>(writeHoriz, d_writeHoriz, cudaMemcpyHostToDevice);

	gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRows * nCols * sizeof(F)));

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	cuReshapeCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, d_nRows, d_nCols, d_nRowsR, d_nColsR, d_readHoriz, d_writeHoriz);

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowsR));
	gpuErrchk(cudaFree(d_nColsR));
	gpuErrchk(cudaFree(d_readHoriz));
	gpuErrchk(cudaFree(d_writeHoriz));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	reshapeTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* reshapeCxMat<float>(float*&, long, long, long, long, bool, bool);
template double* reshapeCxMat<double>(double*&, long, long, long, long, bool, bool);

template<typename F>
F* submatCxMat(F*& m, long nCols, long r0, long c0, long rf, long cf) {

	std::clock_t start;
	start = std::clock();

	long* d_nCols;
	long* d_nRowSub;
	long* d_nColSub;
	long* d_r0;
	long* d_c0;

	long nColSub = cf - c0 + 1;
	long nRowSub = rf - r0 + 1;

	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nRowSub, d_nRowSub, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nColSub, d_nColSub, cudaMemcpyHostToDevice);
	alloCopyValue<long>(r0, d_r0, cudaMemcpyHostToDevice);
	alloCopyValue<long>(c0, d_c0, cudaMemcpyHostToDevice);

	F* d_p;
	gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRowSub * nColSub * sizeof(F)));

	std::vector<dim3> kDims = matrixKernelDims(nRowSub, nColSub);

	cuSubmatCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, d_nCols, d_nRowSub, d_nColSub, d_r0, d_c0);

	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowSub));
	gpuErrchk(cudaFree(d_nColSub));
	gpuErrchk(cudaFree(d_r0));
	gpuErrchk(cudaFree(d_c0));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	submatTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* submatCxMat<float>(float*&, long, long, long, long, long);
template double* submatCxMat<double>(double*&, long, long, long, long, long);

template<typename F>
F* deckExpansionCxMat(F*& m, long nRows, long nCols, long deckSize, bool horizontalExpand) {

	std::clock_t start;
	start = std::clock();

	long nRowsExp = nRows;
	long nColsExp = nCols;

	if (horizontalExpand) nColsExp = deckSize * nCols;
	else nRowsExp = deckSize * nRows;

	long* d_nRows;
	long* d_nCols;
	long* d_deckSize;
	long* d_nRowsExp;
	long* d_nColsExp;
	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nRowsExp, d_nRowsExp, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nColsExp, d_nColsExp, cudaMemcpyHostToDevice);
	alloCopyValue<long>(deckSize, d_deckSize, cudaMemcpyHostToDevice);

	F* d_p;
	gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRowsExp * nColsExp * sizeof(F)));

	std::vector<dim3> kDims = matrixKernelDims(nRowsExp, nColsExp);
	cuDeckExpansionCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, d_nRows, d_nCols, d_nColsExp, d_nRowsExp);

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowsExp));
	gpuErrchk(cudaFree(d_nColsExp));
	gpuErrchk(cudaFree(d_deckSize));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	deckExpansionTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* deckExpansionCxMat<float>(float*&, long, long, long, bool);
template double* deckExpansionCxMat<double>(double*&, long, long, long, bool);

template<typename F>
F* joinCxMat(F*& m, long nRows, long nCols, F*& s, long nRowSub, long nColSub, bool horizJoin) {

	std::clock_t start;
	start = std::clock();

	F* d_p;

	long* d_nRows;
	long* d_nCols;
	long* d_joinSize;

	if (horizJoin) {

		alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
		alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
		alloCopyValue<long>(nColSub, d_joinSize, cudaMemcpyHostToDevice);

		gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRows * (nCols + nColSub) * sizeof(F)));
		std::vector<dim3> kDims = matrixKernelDims(nRows, nCols + nColSub);

		cuJoinHorizontalCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, s, d_nRows, d_nCols, d_joinSize);

	} else {

		alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
		alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
		alloCopyValue<long>(nRowSub, d_joinSize, cudaMemcpyHostToDevice);

		gpuErrchk(cudaMalloc((void**) &d_p, 2 * (nRows + nRowSub) * nCols * sizeof(F)));
		std::vector<dim3> kDims = matrixKernelDims(nRows + nRowSub, nCols);

		cuJoinVerticalCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, s, d_nRows, d_nCols, d_joinSize);
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_joinSize));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	joinTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* joinCxMat<float>(float*&, long, long, float*&, long, long, bool);
template double* joinCxMat<double>(double*&, long, long, double*&, long, long, bool);

template<typename F>
F* fillCxMat(F*& m, long nRows, long nCols, F realVal, F imagVal) {

	std::clock_t start;
	start = std::clock();

	long* d_nRows;
	long* d_nCols;
	F* d_realVal;
	F* d_imagVal;

	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<F>(realVal, d_realVal, cudaMemcpyHostToDevice);
	alloCopyValue<F>(imagVal, d_imagVal, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	cuFillCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_realVal, d_imagVal);

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_realVal));
	gpuErrchk(cudaFree(d_imagVal));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return m;
}
template float* fillCxMat<float>(float*&, long, long, float, float);
template double* fillCxMat<double>(double*&, long, long, double, double);

// Horizontal reduction is much faster due to coalesced memory access
template<typename F, typename uInt>
F* dimensionReductionCxMat(F*& m, uInt nRows, uInt nCols, bool reduceRows) {

	std::clock_t start;
	start = std::clock();

	F* d_p;

	uInt reductionSize;
	if (reduceRows) {
		reductionSize = 2 * nCols;
	} else {
		reductionSize = nRows;
	}

	unsigned int threadsPerBlock;
	for (unsigned int i = MAX_BLOCK_DIM; i > 0; i /= 2) {
		if (reductionSize > i) {
			threadsPerBlock = i;
			break;
		}
	}

	uInt* d_reductionSize;

	alloCopyValue<uInt>(reductionSize, d_reductionSize, cudaMemcpyHostToDevice);

	if (reduceRows) {

		gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRows * sizeof(F)));

		dim3 gridDims (nRows, 1);
		dim3 blockDims(1, threadsPerBlock);

		size_t smemSize = 2 * threadsPerBlock * sizeof(F);

		/*printf("nElements: %ld\n", nElements);
		printf("gridDims:  %ld x %ld\n", gridDims.x, gridDims.y);
		printf("blockDims: %ld x %ld\n", blockDims.x, blockDims.y);
		printf("threadsPerBlock: %u\n", threadsPerBlock);
		printf("smemSize: %ld\n", smemSize);*/

		switch (threadsPerBlock) {
		case 1024:
			rowReductionCxMat<F,uInt,1024><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 512:
			rowReductionCxMat<F,uInt, 512><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 256:
			rowReductionCxMat<F,uInt, 256><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 128:
			rowReductionCxMat<F,uInt, 128><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 64:
			rowReductionCxMat<F,uInt,  64><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 32:
			rowReductionCxMat<F,uInt,  32><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case 16:
			rowReductionCxMat<F,uInt,  16><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  8:
			rowReductionCxMat<F,uInt,   8><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  4:
			rowReductionCxMat<F,uInt,   4><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  2:
			rowReductionCxMat<F,uInt,   2><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		}
	} else {

		gpuErrchk(cudaMalloc((void**) &d_p, 2 * nCols * sizeof(F)));

		dim3 gridDims (2 * nCols, 1);
		dim3 blockDims(1, threadsPerBlock);

		size_t smemSize = 2 * threadsPerBlock * sizeof(F);

		switch (threadsPerBlock) {
		case 1024:
			columnReductionCxMat<F,uInt,1024><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  512:
			columnReductionCxMat<F,uInt, 512><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  256:
			columnReductionCxMat<F,uInt, 256><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case  128:
			columnReductionCxMat<F,uInt, 128><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case   64:
			columnReductionCxMat<F,uInt,  64><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case   32:
			columnReductionCxMat<F,uInt,  32><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case   16:
			columnReductionCxMat<F,uInt,  16><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case    8:
			columnReductionCxMat<F,uInt,   8><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case    4:
			columnReductionCxMat<F,uInt,   4><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		case    2:
			columnReductionCxMat<F,uInt,   2><<<gridDims,blockDims,smemSize>>>(d_p, m, d_reductionSize); break;
		}
	}

	gpuErrchk(cudaFree(d_reductionSize));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	reductionTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float*  dimensionReductionCxMat<float,unsigned int>(float*&, unsigned int, unsigned int, bool);
template double* dimensionReductionCxMat<double,unsigned int>(double*&, unsigned int, unsigned int, bool);
template float*  dimensionReductionCxMat<float,unsigned long>(float*&, unsigned long, unsigned long, bool);
template double* dimensionReductionCxMat<double,unsigned long>(double*&, unsigned long, unsigned long, bool);

template<typename F>
F* transposeCxMat(F*& m, long nRows, long nCols) {

	std::clock_t start;
	start = std::clock();

	F* d_p;

	long* d_nRows;
	long* d_nCols;
	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);

	int tilesPerBlock = 1; //  how many tiles (in one dimension) will each thread block handle?
	long maxBlockSize = MAX_BLOCK_SIZE;
	long horizontalBlocks = nCols / (maxBlockSize * tilesPerBlock);
	if (nCols % (maxBlockSize * tilesPerBlock) != 0) horizontalBlocks += 1;
	long verticalBlocks = nRows / maxBlockSize;
	if (nRows % maxBlockSize != 0) verticalBlocks += 1;

	gpuErrchk(cudaMalloc((void**) &d_p, 2 * nRows * nCols * sizeof(F)));

	dim3 gridDims(horizontalBlocks, verticalBlocks);
	dim3 blockDims(maxBlockSize, maxBlockSize);

	switch (tilesPerBlock) {
	case 16:
		cuTransposeCxMat<F, 16><<<gridDims, blockDims>>>(d_p, m, d_nRows, d_nCols); break;
	case 8:
		cuTransposeCxMat<F, 8><<<gridDims, blockDims>>>(d_p, m, d_nRows, d_nCols); break;
	case 4:
		cuTransposeCxMat<F, 4><<<gridDims, blockDims>>>(d_p, m, d_nRows, d_nCols); break;
	case 2:
		cuTransposeCxMat<F, 2><<<gridDims, blockDims>>>(d_p, m, d_nRows, d_nCols); break;
	case 1:
		cuTransposeCxMat<F, 1><<<gridDims, blockDims>>>(d_p, m, d_nRows, d_nCols); break;
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	transposeTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* transposeCxMat<float>(float*& m, long nRows, long nCols);
template double* transposeCxMat<double>(double*& m, long nRows, long nCols);

template<typename T>
__global__ void setArrayAt(T* array, long index, T val) {
	array[index] = val;
}

template<typename F,typename uInt>
void setByTiledPredicateCxMat(F*& pred, uInt nRowsP, uInt nColsP, Inequality eq, F testVal, F setVal, std::vector<F*>& M, uInt nRows, uInt nCols) {

	std::clock_t start;
	start = std::clock();

	uInt* d_nRowsP;
	uInt* d_nColsP;
	uInt* d_nRows;
	uInt* d_nCols;
	F* d_testVal;
	F* d_setVal;
	alloCopyValue<uInt>(nRowsP, d_nRowsP, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nColsP, d_nColsP, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<uInt>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<F>(testVal, d_testVal, cudaMemcpyHostToDevice);
	alloCopyValue<F>(setVal, d_setVal, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	uInt nMatrices = M.size();
	F** mArray;
	gpuErrchk(cudaMalloc((void**) &mArray, nMatrices * sizeof(F*)));

	for (int i = 0; i < M.size(); i++) {
		setArrayAt<F*><<<1, 1>>>(mArray, i, M.at(i));
	}

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (eq == GreaterThan) {
		switch (nMatrices) {
		case 1:
		cuTiledIsGreaterThan<F,uInt,1><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 2:
		cuTiledIsGreaterThan<F,uInt,2><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 3:
		cuTiledIsGreaterThan<F,uInt,3><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		}
	} else if (eq == LessThan) {
		switch (nMatrices) {
		case 1:
		cuTiledIsLessThan<F,uInt,1><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 2:
		cuTiledIsLessThan<F,uInt,2><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 3:
		cuTiledIsLessThan<F,uInt,3><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		}
	} else if (eq == NotEqualTo) {
		switch (nMatrices) {
		case 1:
		cuTiledIsNotEqualTo<F,uInt,1><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 2:
		cuTiledIsNotEqualTo<F,uInt,2><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 3:
		cuTiledIsNotEqualTo<F,uInt,3><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		}
	} else if (eq == EqualTo) {
		switch (nMatrices) {
		case 1:
		cuTiledIsEqualTo<F,uInt,1><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 2:
		cuTiledIsEqualTo<F,uInt,2><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		case 3:
		cuTiledIsEqualTo<F,uInt,3><<<kDims[0], kDims[1]>>>(pred, d_nRowsP, d_nColsP, d_testVal, d_setVal, d_nRows, d_nCols, mArray); break;
		}
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_nRowsP));
	gpuErrchk(cudaFree(d_nColsP));
	gpuErrchk(cudaFree(d_testVal));
	gpuErrchk(cudaFree(d_setVal));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	inequalityTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;
}
template void setByTiledPredicateCxMat<float,unsigned int>(float*&, unsigned int, unsigned int, Inequality, float, float, std::vector<float*>&, unsigned int, unsigned int);
template void setByTiledPredicateCxMat<double,unsigned int>(double*&, unsigned int, unsigned int, Inequality, double, double, std::vector<double*>&, unsigned int, unsigned int);
template void setByTiledPredicateCxMat<float,unsigned long>(float*&, unsigned long, unsigned long, Inequality, float, float, std::vector<float*>&, unsigned long, unsigned long);
template void setByTiledPredicateCxMat<double,unsigned long>(double*&, unsigned long, unsigned long, Inequality, double, double, std::vector<double*>&, unsigned long, unsigned long);

template<typename F>
F* tiledDiagonalCxMat(F*& m, long nRows, long nCols, long tileSize, bool horizontalCompress) {

	std::clock_t start;
	start = std::clock();

	F* d_p;

	long elementsPerTile = tileSize;
	long horizontalTiles = nCols / tileSize;
	long verticalTiles = nRows / tileSize;
	long nElements = elementsPerTile * verticalTiles * horizontalTiles;

	long xTileSize;
	long yTileSize;

	if (horizontalCompress) {
		xTileSize = tileSize;
		yTileSize = 1;
	} else {
		xTileSize = 1;
		yTileSize = tileSize;
	}

	long* d_nRows;
	long* d_nCols;
	long* d_xTileSize;
	long* d_yTileSize;
	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<long>(xTileSize, d_xTileSize, cudaMemcpyHostToDevice);
	alloCopyValue<long>(yTileSize, d_yTileSize, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_p, 2 * nElements * sizeof(F));
	std::vector<dim3> kDims = matrixKernelDims(nRows / yTileSize, nCols / xTileSize);

	cuTiledDiagonalCxMat<F><<<kDims[0], kDims[1]>>>(d_p, m, d_nRows, d_nCols, d_xTileSize, d_yTileSize);

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_xTileSize));
	gpuErrchk(cudaFree(d_yTileSize));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	tiledDiagonalTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	return d_p;
}
template float* tiledDiagonalCxMat<float>(float*&, long, long, long, bool);
template double* tiledDiagonalCxMat<double>(double*&, long, long, long, bool);

template<typename F>
bool globalPredicateCxMat(F*& m, long nRows, long nCols, Inequality eq, F val) {

	std::clock_t start;
	start = std::clock();

	unsigned int isTrue = 1;

	long* d_nRows;
	long* d_nCols;
	F* d_val;
	unsigned int* d_isTrue;
	alloCopyValue<long>(nRows, d_nRows, cudaMemcpyHostToDevice);
	alloCopyValue<long>(nCols, d_nCols, cudaMemcpyHostToDevice);
	alloCopyValue<F>(val, d_val, cudaMemcpyHostToDevice);
	alloCopyValue<unsigned int>(isTrue, d_isTrue, cudaMemcpyHostToDevice);

	std::vector<dim3> kDims = matrixKernelDims(nRows, nCols);

	if (eq == EqualTo) {
		cuGlobalEqualToCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_val, d_isTrue);
	} else if (eq == NotEqualTo) {
		cuGlobalNotEqualToCxMat<F><<<kDims[0], kDims[1]>>>(m, d_nRows, d_nCols, d_val, d_isTrue);
	}

	gpuErrchk(cudaFree(d_nRows));
	gpuErrchk(cudaFree(d_nCols));
	gpuErrchk(cudaFree(d_val));
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	alloCopyValue(isTrue, d_isTrue, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaFree(d_isTrue));

	globalPredicateTime += (std::clock() - start) / (double) CLOCKS_PER_SEC;

	if (isTrue == 1) return true;
	else return false;
}
template bool globalPredicateCxMat<float>(float*& m, long nRows, long nCols, Inequality eq, float val);
template bool globalPredicateCxMat<double>(double*& m, long nRows, long nCols, Inequality eq, double val);

std::vector<dim3> matrixKernelDims(long nRows, long nCols) {

	long blockWidth = MAX_BLOCK_SIZE;
	long xBlocks = nCols / blockWidth;
	if (nCols % blockWidth != 0) xBlocks++;
	long yBlocks = nRows / blockWidth;
	if (nRows % blockWidth != 0) yBlocks++;

	std::vector<dim3> d;
	d.push_back(dim3(xBlocks, yBlocks, 1));
	d.push_back(dim3(blockWidth, blockWidth, 1));

	return d;
}

template<typename F>
void printCxMat(F*& m, long nRows, long nCols) {
	long k;
	for (long i = 0; i < nRows; i++) {
		for (long j = 0; j < nCols; j++) {
			k = 2 * (j + i * nCols);
			printf("[%8.2e, %8.2ei] ", m[k], m[k + 1]);
		}
		printf("\n");
	}
	printf("\n");
}
template void printCxMat<float>(float*&, long, long);
template void printCxMat<double>(double*&, long, long);

void profileCuFunctions(double duration) {
	std::vector<double> times = {tiledOpTime, elementOpTime, reductionTime, inequalityTime, cardExpansionTime, reshapeTime,
			submatTime, deckExpansionTime, joinTime, fillTime, transposeTime, alloCopyTime, tiledDiagonalTime, globalPredicateTime};
	std::vector<std::string> funcNames = {"tiledOpTime", "elementOpTime", "reductionTime", "inequalityTime", "cardExpansionTime", "reshapeTime",
			"submatTime", "deckExpansionTime", "joinTime", "fillTime", "transposeTime", "alloCopyTime", "tiledDiagonalTime", "globalPredicateTime"};

	double cudaTime = 0;
	for (int i = 0; i < times.size(); i++) {
		cudaTime += times.at(i);
	}

	double externalTime = duration - cudaTime;

	double maxPercent = 0;
	std::vector<double> timePercent;
	for (int i = 0; i < times.size(); i++) {
		double percentage = 100.0 * times.at(i) / duration;
		timePercent.push_back(percentage);
		if (percentage > maxPercent) maxPercent = percentage;
	}

	int maxTitleSize = 0;
	for (int i = 0; i < funcNames.size(); i++) {
		int length = funcNames.at(i).length();
		if (length > maxTitleSize) {
			maxTitleSize = length;
		}
	}
	printf("\nPercent share of time:\n");
	for (int i = 0; i < times.size(); i++) {
		printf("%s ", funcNames.at(i).c_str());
		for (int j = 0; j < (maxTitleSize - funcNames.at(i).length()); j++) printf(" ");
		printf("|");
		int barSize = 40;
		double percentPerTick = maxPercent / (double) barSize;
		for (int j = 0; j < barSize; j++) {
			if (timePercent.at(i) > ((double) j * percentPerTick)) printf("#");
			else printf(" ");
		}
		printf(" | ");
		printf("%4.2lf (s)\n", times.at(i));
	}
	printf("%4.2lf (s) --> Cuda Time\n", cudaTime);
	printf("%4.2lf (s) --> External Time\n", externalTime);
	printf("%4.2lf (s) --> Total Time\n", duration);
}

void resetCudaTimes() {
	tiledOpTime = 0;
	elementOpTime = 0;
	reductionTime = 0;
	inequalityTime = 0;
	cardExpansionTime = 0;
	reshapeTime = 0;
	submatTime = 0;
	deckExpansionTime = 0;
	joinTime = 0;
	fillTime = 0;
	alloCopyTime = 0;
	transposeTime = 0;
	tiledDiagonalTime = 0;
	globalPredicateTime = 0;
}

void queryDeviceInfo() {
    printf("CUDA Device Query...\n");

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

void checkMemoryUsage() {
	size_t mem_free = 0;
	size_t mem_tot = 0;
	cudaSetDevice(0);
	cudaMemGetInfo(&mem_free, &mem_tot);
	long memUsage = mem_tot - mem_free;

	if (memUsage > maxMemoryUsage) maxMemoryUsage = memUsage;
}

double getMaxMemoryUsage() {
	return ((double) maxMemoryUsage) / (1000000.0);
}

void freeGlobalMemory() {
	cudaThreadExit();
	cudaDeviceReset();
}

long getCoreCount(int deviceNumber) {

	cudaDeviceProp devProp;
    gpuErrchk(cudaGetDeviceProperties(&devProp, deviceNumber));
    int majorRev = devProp.major;
    int minorRev = devProp.minor;
    double nMPs = (double) devProp.multiProcessorCount;
    double coresPerMP;
    if      (majorRev == 1) coresPerMP = 8.0;
    else if (majorRev == 2) {
    	if  (minorRev <  1) coresPerMP = 32.0;
    	else                coresPerMP = 48.0;
    }
    else if (majorRev == 3) coresPerMP = 192.0;
    else                    coresPerMP = 128.0;
    long nCores = (long) coresPerMP * nMPs;

    return nCores;
}
