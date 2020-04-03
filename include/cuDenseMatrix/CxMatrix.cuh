#include "util.cuh"
#include <vector>
#include "armadillo"
#include <string>
#include <iostream>
#include <ctime>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace cdm { // Cuda Dense Matrix

enum MemLoc {
	Host = 0,
	Device = 1
};

enum Dimension {
	Rows = 0,
	Columns = 1,
	Square = 2
};

template<typename F, typename uInt>
class CxMatrix {
public:

	CxMatrix(arma::cx_mat armaMat) {
		nRows = armaMat.n_rows;
		nCols = armaMat.n_cols;
		loc = Host;
		hostData = new F[2 * nRows * nCols];
		uInt k;
		std::complex<double> armaVal;
		for (uInt i = 0; i < nCols; i++) {
			for (uInt j = 0; j < nRows; j++) {
				k = i + j * nCols;
				armaVal = armaMat(j, i);
				hostData[2 * k + 0] = armaVal.real();
				hostData[2 * k + 1] = armaVal.imag();
			}
		}
		deviceData = new F;
		hostDataAlloc = true;
		deviceDataAlloc = false;
		useFree = false;
	}

	CxMatrix(F*& data, MemLoc _loc, uInt _nRows, uInt _nCols) {
		nRows = _nRows;
		nCols = _nCols;
		loc = _loc;

		if (loc == Host) {
			hostData = data;
			hostDataAlloc = true;
			deviceDataAlloc = false;
		} else {
			deviceData = data;
			hostDataAlloc = false;
			deviceDataAlloc = true;
		}
		useFree = false;
	}

	CxMatrix(uInt _nRows, uInt _nCols, std::complex<F> fillVal) {
		nRows = _nRows;
		nCols = _nCols;
		loc = Device;
		hostData = new F;

		size_t size = 2 * nRows * nCols * sizeof(F);
		gpuErrchk(cudaMalloc((void**) &deviceData, size));
		deviceDataAlloc = true;
		hostDataAlloc = false;
		useFree = false;
		fill(fillVal);
	}

	CxMatrix(const CxMatrix& mat) {

		nRows = mat.numberOfRows();
		nCols = mat.numberOfCols();
		loc = mat.memoryLocation();

		size_t size = 2 * nRows * nCols * sizeof(F);

		useFree = true;

		if (mat.memoryLocation() == Host) {
			hostData = (F*) malloc(size);
			memcpy(hostData, mat.data(), size);
			hostDataAlloc = true;
			deviceDataAlloc = false;
		}
		else {
			gpuErrchk(cudaMalloc((void**) &deviceData, size));
			gpuErrchk(cudaMemcpy(deviceData, mat.data(), size, cudaMemcpyDeviceToDevice));
			deviceDataAlloc = true;
			hostDataAlloc = false;
		}
	}

	CxMatrix() {
		nRows = 0;
		nCols = 0;
		loc = Host;
		hostData = new F;
		deviceData = new F;
		hostDataAlloc = false;
		deviceDataAlloc = false;
		useFree = false;
	}

	arma::cx_mat* toArmaMat() { // TODO: copy by array
		copyTo(Host);
		arma::cx_mat* armaMat = new arma::cx_mat(nRows, nCols);

		uInt k;
		for (uInt i = 0; i < nRows; i++) {
			for (uInt j = 0; j < nCols; j++) {
				k = j + i * nCols;
				armaMat->at(i, j) = std::complex<F>(hostData[2 * k + 0], hostData[2 * k + 1]);
			}
		}
		return armaMat;
	}

	void fill(std::complex<F> fillVal) {
		copyTo(Device);

		deviceData = fillCxMat<F>(deviceData, nRows, nCols, fillVal.real(), fillVal.imag());
	}

	CxMatrix<F,uInt>* fillAsCopy(std::complex<F> fillVal) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->fill(fillVal);
		return copyMat;
	}

	CxMatrix<F,uInt>* col(uInt colNumber) {
		return submatAsCopy(0, colNumber, nRows - 1, colNumber);
	}

	CxMatrix<F,uInt>* colRange(uInt startCol, uInt endCol) {
		return submatAsCopy(0, startCol, nRows - 1, endCol);
	}

	CxMatrix<F,uInt>* row(uInt rowNumber) {
		return submatAsCopy(rowNumber, 0, rowNumber, nCols - 1);
	}

	CxMatrix<F,uInt>* rowRange(uInt startRow, uInt endRow) {
		return submatAsCopy(startRow, 0, endRow, nCols - 1);
	}

	CxMatrix<F,uInt>* clone() {
		return this;
	}

	CxMatrix<F,uInt>* copy() {
		return new CxMatrix<F,uInt>(*this);
	}

	uInt numberOfRows() const {
		return nRows;
	}

	uInt numberOfCols() const {
		return nCols;
	}

	MemLoc memoryLocation() const {
		return loc;
	}

	F* data() const {
		if (loc == Host) return hostData;
		else return deviceData;
	};


	void copyTo(MemLoc dest) {
		if (dest == Host && loc == Device) {
			alloCopyData(cudaMemcpyDeviceToHost);
			freeMem(Device);
			loc = Host;
			hostDataAlloc = true;
		}
		if (dest == Device && loc == Host) {
			alloCopyData(cudaMemcpyHostToDevice);
			freeMem(Host);
			loc = Device;
			deviceDataAlloc = true;
		}
	}

	void freeMem() {
		freeMem(Host);
		freeMem(Device);
	}

	void freeMem(MemLoc targLoc) {
		if (targLoc == Host && hostDataAlloc) {
			if (useFree) free(hostData);
			else delete[] (F*) hostData;
			hostDataAlloc = false;
		}
		if (targLoc == Device && deviceDataAlloc) {
			gpuErrchk(cudaFree(deviceData));
			deviceDataAlloc = false;
		}
	}

	bool isAlloc(MemLoc loc) {
		if (loc == Host) return hostDataAlloc;
		else return deviceDataAlloc;
	}

	void submat(uInt r0, uInt c0, uInt rf, uInt cf) {
		copyTo(Device);

		uInt nRowSub = rf - r0 + 1;
		uInt nColSub = cf - c0 + 1;

		F* tempDeviceData = submatCxMat<F>(deviceData, nCols, r0, c0, rf, cf);
		gpuErrchk(cudaFree(deviceData));
		deviceData = tempDeviceData;

		nRows = nRowSub;
		nCols = nColSub;
	}

	CxMatrix<F,uInt>* submatAsCopy(uInt r0, uInt c0, uInt rf, uInt cf) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->submat(r0, c0, rf, cf);
		return copyMat;
	}

	template<typename T>
	void elementOp(CxOperation op, T scalar) {
		copyTo(Device);
		T* scalarPtr = &scalar;
		deviceData = elementOpCxMat<F,T,uInt>(deviceData, nRows, nCols, scalarPtr, op);
	}

	template<typename T>
	CxMatrix<F,uInt>* elementOpAsCopy(CxOperation op, T scalar) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->elementOp<T>(op, scalar);
		return copyMat;
	}

	template<typename T>
	void elementOp(CxOperation op, std::complex<T> scalar) {
		copyTo(Device);
		F* cxScalar = new F[2];
		cxScalar[0] = scalar.real();
		cxScalar[1] = scalar.imag();
		deviceData = elementOpCxMat<F,T,uInt>(deviceData, nRows, nCols, cxScalar, op);
	}

	template<typename T>
	CxMatrix<F,uInt>* elementOpAsCopy(CxOperation op, std::complex<F> scalar) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->elementOp<T>(op, scalar);
		return copyMat;
	}

	void multiply(std::complex<F> scalar) {
		elementOp<F>(Multiply, scalar);
	}

	CxMatrix<F,uInt>* multiplyAsCopy(std::complex<F> scalar) {
		return elementOpAsCopy<F>(Multiply, scalar);
	}

	void power(int exponent) {
		elementOp<int>(Power, exponent);
	}

	CxMatrix<F,uInt>* powerAsCopy(int exponent) {
		return elementOpAsCopy<int>(Power, exponent);
	}

	void realSqrt() {
		elementOp<int>(RealSqrt, -1);
	}

	CxMatrix<F,uInt>* realSqrtAsCopy() {
		return elementOpAsCopy<int>(RealSqrt, -1);
	}

	void exp() {
		elementOp<int>(Exp, -1);
	}

	CxMatrix<F,uInt>* expAsCopy() {
		return elementOpAsCopy<int>(Exp, -1);
	}

	void add(std::complex<double> val) {
		elementOp<F>(Sum, val);
	}

	CxMatrix<F,uInt>* addAsCopy(std::complex<double> val) {
		return elementOpAsCopy<F>(Sum, val);
	}

	void tiledOperation(CxOperation op, CxMatrix<F,uInt>* t) {
		copyTo(Device);
		t->copyTo(Device);

		uInt nRowsT = t->numberOfRows();
		uInt nColsT = t->numberOfCols();
		F* tileData = t->data();

		deviceData = tiledOperationCxMat<F>(deviceData, nRows, nCols, tileData, nRowsT, nColsT, op);
	}

	CxMatrix<F,uInt>* tiledOperationAsCopy(CxOperation op, CxMatrix<F,uInt>* t) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix(*this);
		copyMat->tiledOperation(op, t);
		return copyMat;
	}

	void multiply(CxMatrix<F,uInt>* t) {
		tiledOperation(Multiply, t);
	}

	CxMatrix<F,uInt>* multiplyAsCopy(CxMatrix<F,uInt>* t) {
		return tiledOperationAsCopy(Multiply, t);
	}

	void add(CxMatrix<F,uInt>* t) {
		tiledOperation(Sum, t);
	}

	CxMatrix<F,uInt>* addAsCopy(CxMatrix<F,uInt>* t) {
		return tiledOperationAsCopy(Sum, t);
	}

	void realValueInequality(Inequality eq, F realVal) {
		copyTo(Device);
		deviceData = inequalityCxMat<F>(deviceData, nRows, nCols, realVal, eq);
	}

	CxMatrix<F,uInt>* realValueInequalityAsCopy(Inequality eq, F realVal) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->realValueInequality(eq, realVal);
		return copyMat;
	}

	void cardExpansion(Dimension expansionDim, uInt cardSize) {
		copyTo(Device);

		if (expansionDim == Columns) {
			F* tempDeviceData = cardExpansionCxMat<F,uInt>(deviceData, nRows, nCols, cardSize, true);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nCols *= cardSize;
		} else {
			F* tempDeviceData = cardExpansionCxMat<F,uInt>(deviceData, nRows, nCols, cardSize, false);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nRows *= cardSize;
		}
	}

	CxMatrix<F,uInt>* cardExpansionAsCopy(Dimension expansionDim, uInt cardSize) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->cardExpansion(expansionDim, cardSize);
		return copyMat;
	}

	void deckExpansion(Dimension expansionDim, uInt deckSize) {
		useFree = false;
		copyTo(Device);

		if (expansionDim == Columns) {
			F* tempDeviceData = deckExpansionCxMat(deviceData, nRows, nCols, deckSize, true);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nCols *= deckSize;
		} else {
			F* tempDeviceData = deckExpansionCxMat(deviceData, nRows, nCols, deckSize, false);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nRows *= deckSize;
		}
	}

	CxMatrix<F,uInt>* deckExpansionAsCopy(Dimension expansionDim, uInt deckSize) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->deckExpansion(expansionDim, deckSize);
		return copyMat;
	}

	void reshape(uInt nRowsR, uInt nColsR, Dimension readDirection, Dimension writeDirection) {
		copyTo(Device);

		bool readHoriz  = (readDirection == Columns);
		bool writeHoriz = (writeDirection == Columns);

		if (!(readHoriz && writeHoriz)) { // If read and write is horiz, then no GPU func needed
			F* tempDeviceData = reshapeCxMat<F>(deviceData, nRows, nCols, nRowsR, nColsR, readHoriz, writeHoriz);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
		}

		nRows = nRowsR;
		nCols = nColsR;
	}

	CxMatrix<F,uInt>* reshapeAsCopy(uInt nRowsR, uInt nColsR, Dimension readDirection, Dimension writeDirection) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->reshape(nRowsR, nColsR, readDirection, writeDirection);
		return copyMat;
	}

	void join(CxMatrix<F,uInt>* t, Dimension joinDirection) {
		copyTo(Device);
		t->copyTo(Device);
		F* tDeviceData = t->data();

		uInt nRowsT = t->numberOfRows();
		uInt nColsT = t->numberOfCols();

		if (joinDirection == Columns) {
			F* tempDeviceData = joinCxMat<F>(deviceData, nRows, nCols, tDeviceData, nRowsT, nColsT, true);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nCols += nColsT;
		} else {
			F* tempDeviceData = joinCxMat<F>(deviceData, nRows, nCols, tDeviceData, nRowsT, nColsT, false);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nRows += nRowsT;
		}
	}

	CxMatrix<F,uInt>* joinAsCopy(CxMatrix<F,uInt>* t, Dimension joinDirection) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->join(t, joinDirection);
		return copyMat;
	}

	void transpose() {
		copyTo(Device);

		uInt _nRows = nRows;
		uInt _nCols = nCols;

		F* tempDeviceData = transposeCxMat<F>(deviceData, nRows, nCols);
		gpuErrchk(cudaFree(deviceData));
		deviceData = tempDeviceData;

		nRows = _nCols;
		nCols = _nRows;
	}

	CxMatrix<F,uInt>* transposeAsCopy() {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->transpose();
		return copyMat;
	}

	void reduceBy(Dimension reductionDimension) {
		copyTo(Device);

		if (reductionDimension == Rows) {
			F* tempDeviceData = dimensionReductionCxMat<F,uInt>(deviceData, nRows, nCols, true);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nCols = 1;
		} else {
			F* tempDeviceData = dimensionReductionCxMat<F,uInt>(deviceData, nRows, nCols, false);
			gpuErrchk(cudaFree(deviceData));
			deviceData = tempDeviceData;
			nRows = 1;
		}
	}

	void cardReduction(Dimension reductionDim, uInt cardSize) {
		copyTo(Device);

		uInt _nRows = nRows;
		uInt _nCols = nCols;

		if (reductionDim == Rows) {
			uInt nCards = nCols / cardSize;
			/*printf("cardSize: %u\n", cardSize);
			printf("nCols: %u\n", nCols);
			printf("nCards: %u\n", nCards);*/
			reshape(_nRows * nCards, cardSize, Columns, Columns);
			/*printf("cardSize: %u\n", cardSize);
			printf("nCols: %u\n", nCols);
		    printf("nCards: %u\n", nCards);*/
			reduceBy(Rows);
			reshape(_nRows, nCards, Columns, Columns);
		} else {
			transpose();
			cardReduction(Rows, cardSize);
			transpose();
		}
	}

	CxMatrix<F,uInt>* cardReductionAsCopy(Dimension reductionDimension, uInt cardSize) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->cardReduction(reductionDimension, cardSize);
		return copyMat;
	}

	void deckReduction(Dimension reductionDim, uInt cardSize) {
		copyTo(Device);

		uInt _nRows = nRows;
		uInt _nCols = nCols;
		uInt nCards;

		if (reductionDim == Rows) {
			transpose();
			deckReduction(Columns, cardSize);
			transpose();

		} else { // Vertical reduce
			nCards = _nRows / cardSize;
			reshape(nCards, cardSize * _nCols, Columns, Columns);
			reduceBy(Columns);
			reshape(cardSize, _nCols, Columns, Columns);
		}
	}

	void tiledPredicate(Inequality eq, F testVal, std::vector<CxMatrix<F,uInt>*>& targetMats, F setVal) {
		copyTo(Device);

		std::vector<F*> targetArrays;
		for (int i = 0; i < targetMats.size(); i++) {
			CxMatrix<F,uInt>* targetMat = targetMats.at(i);
			targetMat->copyTo(Device);
			targetArrays.push_back(targetMat->data());
		}

		uInt target_nRows = targetMats.at(0)->numberOfRows();
		uInt target_nCols = targetMats.at(0)->numberOfCols();

		setByTiledPredicateCxMat<F,uInt>(deviceData, nRows, nCols, eq, testVal, setVal, targetArrays, target_nRows, target_nCols);
	}

	void tiledDiagonal(Dimension reductionDim, uInt tileSize) {
		copyTo(Device);

		bool horizontalCompress = (reductionDim == Columns);

		F* tempDeviceData = tiledDiagonalCxMat<F>(deviceData, nRows, nCols, tileSize, horizontalCompress);
		gpuErrchk(cudaFree(deviceData));
		deviceData = tempDeviceData;

		if (horizontalCompress) nCols /= tileSize;
		else nRows /= tileSize;
	}

	CxMatrix<F,uInt>* tiledDiagonalAsCopy(Dimension reductionDim, uInt tileSize) {
		CxMatrix<F,uInt>* copyMat = new CxMatrix<F,uInt>(*this);
		copyMat->tiledDiagonal(reductionDim, tileSize);
		return copyMat;
	}

	bool globalPredicate(Inequality eq, F val) {
		copyTo(Device);

		return globalPredicateCxMat(deviceData, nRows, nCols, eq, val);
	}

	void printSize(std::string title = "") {
		std::cout << "\n" << title << "\n";
		printf("nRows: %ld, nCols: %ld\n", nRows, nCols);
	}

	void print(std::string title = "") {
		copyTo(Host);

		std::cout << "\n" << title << "\n";
		printSize();

		uInt k;
		for (uInt i = 0; i < nRows; i++) {
			for (uInt j = 0; j < nCols; j++) {
				k = 2 * (j + i * nCols);
				printf("[%8.3e, %8.3ei] ", hostData[k], hostData[k + 1]);
			}
			printf("\n");
		}
		printf("\n");
	}

	// Package  1x9 data into 3x3 data
	void package() {
		copyTo(Host);

		F* packageArray = new F[2 * nRows * nCols];

		for (uInt i = 0; i < nRows / 9; i++) {
			for (uInt j = 0; j < nCols; j++) {

				uInt xGrid = j;
				uInt yGrid = i;

				for (uInt k = 0; k < 3; k++) {
					for (uInt l = 0; l < 3; l++) {

						uInt xIdxDest = l + 3 * xGrid;
						uInt yIdxDest = k + 3 * yGrid;
						uInt destIdx = xIdxDest + yIdxDest * (nCols * 3);

						uInt xIdxPull = xGrid;
						uInt yIdxPull = l + 3 * k + 9 * yGrid;
						uInt pullIdx = xIdxPull + yIdxPull * nCols;

						packageArray[2 * destIdx + 0] = hostData[2 * pullIdx + 0];
						packageArray[2 * destIdx + 1] = hostData[2 * pullIdx + 1];
					}
				}
			}
		}
		if (useFree) free (hostData);
		else delete[] (F*) hostData;
		hostData = packageArray;
		nRows /= 3;
		nCols *= 3;
	}

	// unroll data in 3x3 to 1x9
	void unpackage() {
		copyTo(Host);

		F* stripArray = new F[2 * nRows * nCols];

		for (uInt i = 0; i < nRows / 3; i++) {
			for (uInt j = 0; j < nCols / 3; j++) {

				uInt xGrid = j;
				uInt yGrid = i;

				for (uInt k = 0; k < 3; k++) {
					for (uInt l = 0; l < 3; l++) {

						uInt xIdxDest = xGrid;
						uInt yIdxDest = l + 3 * k + 9 * yGrid;
						uInt destIdx = xIdxDest + yIdxDest * (nCols / 3);

						uInt xIdxPull = l + 3 * xGrid;
						uInt yIdxPull = k + 3 * yGrid;
						uInt pullIdx = xIdxPull + yIdxPull * nCols;

						stripArray[2 * destIdx + 0] = hostData[2 * pullIdx + 0];
						stripArray[2 * destIdx + 1] = hostData[2 * pullIdx + 1];
					}
				}
			}
		}
		if (useFree) free(hostData);
		else delete[] (F*) hostData;
		hostData = stripArray;
		useFree = false;
		nRows *= 3;
		nCols /= 3;
	}

private:

	uInt nRows;
	uInt nCols;
	F* hostData;
	F* deviceData;
	MemLoc loc;
	bool hostDataAlloc;
	bool deviceDataAlloc;
	bool useFree;

	void alloCopyData(cudaMemcpyKind kind) {

		uInt length = 2 * nRows * nCols;

		if (kind == cudaMemcpyHostToDevice) {
			alloCopyArray<F>(hostData, deviceData, length, kind);
			loc = Device;
		}
		else if (kind == cudaMemcpyDeviceToHost) {
			alloCopyArray<F>(hostData, deviceData, length, kind);
			loc = Host;
		}
	}

};

} // End namespace
