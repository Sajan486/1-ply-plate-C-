/*
 * Matrix.cpp
 *
 *  Created on: Sep 12, 2017
 *      Author: lance
 */

#include "Matrix.h"

template<typename T>
vat::Mat<T>::Mat(DiskCache* _cache, ul _nRows, ul _nCols, MemorySize maxBlockSize, Order _order) {

	cache = _cache;
	nRows = _nRows;
	nCols = _nCols;
	order = _order;
	ul nElements = nRows * nCols;

	rowOffset = 0lu;
	colOffset = 0lu;

	ul maxElementsBlock = maxBlockSize.nBytes() / sizeof(T);

	if (maxElementsBlock == 0) {
		printf("Warning: Mat initialized with dimension of zero\n");
	}

	ul blockWidth;
	ul blockHeight;

	ul gridHeight = 1lu;  // By default set these values, unless otherwise specified
	ul gridWidth  = 1lu;

	if (nRows == 1lu) {

		blockHeight = 1lu;
		blockWidth  = nElements;
	}
	else if (nCols == 1lu) {

		blockHeight = nElements;
		blockWidth  = 1lu;
	}
	else {

		ul maxSquareBlockDim = (ul)sqrt((double)maxElementsBlock);

		if (nRows <= nCols) {

			gridHeight = nRows / maxSquareBlockDim;
			if (nRows % maxSquareBlockDim != 0) gridHeight++;
			blockHeight = nRows / gridHeight;
			if (nRows % gridHeight != 0) blockHeight++;

			ul maxHorizBlockDim = maxElementsBlock / blockHeight;

			gridWidth = nCols / maxHorizBlockDim;
			if (nCols % maxHorizBlockDim != 0) gridWidth++;
			blockWidth = nCols / gridWidth;
			if (nCols % gridWidth != 0) blockWidth++;
		}
		else {

			gridWidth = nCols / maxSquareBlockDim;
			if (nCols % maxSquareBlockDim != 0) gridWidth++;
			blockWidth = nCols / gridWidth;
			if (nCols % gridWidth != 0) blockWidth++;

			ul maxVertBlockDim = maxElementsBlock / blockWidth;

			gridHeight = nRows / maxVertBlockDim;
			if (nRows % maxVertBlockDim != 0) gridHeight++;
			blockHeight = nRows / gridHeight;
			if (nRows % gridHeight != 0) blockHeight++;
		}
	}

	GridDims  gridDims (gridWidth,  gridHeight);
	BlockDims blockDims(blockWidth, blockHeight);

	grid = new Grid<T>(cache, gridDims, blockDims, order);
	base = true;
}

template<typename T>
vat::Mat<T>::Mat(const Mat<T>& mat) {

	cache = mat.cache;
	nRows = mat.nRows;
	nCols = mat.nCols;
	order = mat.order;

	rowOffset = mat.rowOffset;
	colOffset = mat.colOffset;

	grid = mat.grid;
	base = mat.base;
}

template<typename T>
vat::Mat<T>::~Mat() {
	if (base) delete grid;
}

template<typename T>
vat::DiskCache* vat::Mat<T>::getCache() {
	return cache;
}

/*
 * GETTERS
 */

template<typename T>
vat::BlockDims vat::Mat<T>::getBlockDims() {
	return grid->getBlockDims();
}

template<typename T>
vat::GridDims vat::Mat<T>::getGridDims() {
	return grid->getGridDims();
}

template<typename T>
T* vat::Mat<T>::subMat(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (checkIndices(startRow, startCol, endRow, endCol)) {

		matrixToGridIdx (startRow, startCol, endRow, endCol);
		return grid->subGrid(startRow, startCol, endRow, endCol);
	}
	else return (T*)(0);
}

template<typename T>
T** vat::Mat<T>::subMat2D(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (checkIndices(startRow, startCol, endRow, endCol)) {

		matrixToGridIdx (startRow, startCol, endRow, endCol);
		return grid->subGrid2D(startRow, startCol, endRow, endCol);
	}
	else return (T**)(0);
}

template<typename T>
T* vat::Mat<T>::rows(ul startRow, ul endRow) {

	return subMat(startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
T** vat::Mat<T>::rows2D(ul startRow, ul endRow) {

	return subMat2D(startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
T* vat::Mat<T>::row(ul rowIdx) {

	return rows(rowIdx, rowIdx);
}

template<typename T>
T* vat::Mat<T>::cols(ul startCol, ul endCol) {

	return subMat(0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
T** vat::Mat<T>::cols2D(ul startCol, ul endCol) {

	return subMat2D(0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
T* vat::Mat<T>::col(ul colIdx) {

	return rows(colIdx, colIdx);
}

template<typename T>
T vat::Mat<T>::element(ul rowIdx, ul colIdx) {
	T val = *subMat(rowIdx, colIdx, rowIdx, colIdx);
	return val;
}

template<typename T>
T vat::Mat<T>::operator()(ul rowIdx, ul colIdx) {

	T val = element(rowIdx, colIdx);
	return val;
}

/*
 * SETTERS
 */

template<typename T>
void vat::Mat<T>::subMat(T* pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (checkIndices(startRow, startCol, endRow, endCol)) {

		matrixToGridIdx (startRow, startCol, endRow, endCol);
		grid->subGrid(pData, startRow, startCol, endRow, endCol);
	}
}

template<typename T>
void vat::Mat<T>::subMat(T** pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (checkIndices(startRow, startCol, endRow, endCol)) {

		matrixToGridIdx (startRow, startCol, endRow, endCol);
		grid->subGrid(pData, startRow, startCol, endRow, endCol);
	}
}

template<typename T>
void vat::Mat<T>::rows(T* pData, ul startRow, ul endRow) {

	subMat(pData, startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
void vat::Mat<T>::rows2D(T** pData, ul startRow, ul endRow) {

	subMat(pData, startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
void vat::Mat<T>::row(T* pData, ul rowIdx) {

	rows(pData, rowIdx, rowIdx);
}

template<typename T>
void vat::Mat<T>::cols(T* pData, ul startCol, ul endCol) {
	subMat(pData, 0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
void vat::Mat<T>::cols2D(T** pData, ul startCol, ul endCol) {

	subMat(pData, 0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
void vat::Mat<T>::col(T* pData, ul col) {

	cols(pData, col, col);
}

template<typename T>
void vat::Mat<T>::element(T val, ul rowIdx, ul colIdx) {

	subMat(&val, rowIdx, colIdx, rowIdx, colIdx);
}

template<typename T>
void vat::Mat<T>::fill(T fillVal) {

	grid->fill(fillVal);
}

template<typename T>
vat::ul vat::Mat<T>::getN_Rows() {
	return nRows;
}

template<typename T>
vat::ul vat::Mat<T>::getN_Cols() {
	return nCols;
}

template<typename T>
vat::Mat<T>* vat::Mat<T>::subView(ul startRow, ul startCol, ul endRow, ul endCol) {

	Mat<T>* mat = new Mat<T>(*this);
	mat->base = false;

	ul _nRows = endRow - startRow + 1lu;
	ul _nCols = endCol - startCol + 1lu;

	mat->setView(startRow, startCol, _nRows, _nCols);
	return mat;
}

template<>
void vat::Mat<double>::print(std::string title) {

	if (title.length() > 0) std::cout << title << ":";
	std::cout << std::endl;

	double** pData = subMat2D(0lu, 0lu, nRows - 1lu, nCols - 1lu);

	int leftBorderSize = floor(log10(nRows - 1lu)) + 1lu;
	int columnWidth = 12;

	printf("%*lu", leftBorderSize + 10, 0lu);
	for (ul col = 1; col < nCols; col++) {
		printf("%*lu", columnWidth, col);
	}   printf("\n\n");

	for (ul row = 0; row < nRows; row++) {

		printf("%*lu ", leftBorderSize, row);
		for (ul col = 0; col < nCols; col++) {

			double val;
			if (order == RowMajor) val = pData[row][col];
			else                   val = pData[col][row];

			if (val >= 0) printf("+");
			printf("%.3e", pData[row][col]);
			if (col != nCols - 1lu) printf(", ");
		}
		printf("\n");
	}
}

template<>
void vat::Mat<float>::print(std::string title) {

	if (title.length() > 0) std::cout << title << ":";
	std::cout << std::endl;

	float** pData = subMat2D(0lu, 0lu, nRows - 1lu, nCols - 1lu);

	int leftBorderSize = floor(log10(nRows - 1lu)) + 1lu;
	int columnWidth = 12;

	printf("%*lu", leftBorderSize + 10, 0lu);
	for (ul col = 1; col < nCols; col++) {
		printf("%*lu", columnWidth, col);
	}   printf("\n\n");

	for (ul row = 0; row < nRows; row++) {

		printf("%*lu ", leftBorderSize, row);
		for (ul col = 0; col < nCols; col++) {

			float val;
			if (order == RowMajor) val = pData[row][col];
			else                   val = pData[col][row];

			if (val >= 0) printf("+");
			printf("%.3e", pData[row][col]);
			if (col != nCols - 1lu) printf(", ");
		}
		printf("\n");
	}
}

template<>
void vat::Mat<std::complex<float>>::print(std::string title) {

	if (title.length() > 0) std::cout << title << ":";
	std::cout << std::endl;

	std::complex<float>** pData = subMat2D(0lu, 0lu, nRows - 1lu, nCols - 1lu);

	int leftBorderSize = floor(log10(nRows - 1lu)) + 1lu;
	int columnWidth = 27lu;

	printf("%*lu", leftBorderSize + 23, 0lu);
	for (ul col = 1; col < nCols; col++) {
		printf("%*lu", columnWidth, col);
	}   printf("\n\n");

	for (ul row = 0; row < nRows; row++) {

		printf("%*lu ", leftBorderSize, row);
		for (ul col = 0; col < nCols; col++) {

			std::complex<float> val;
			if (order == RowMajor) val = pData[row][col];
			else                   val = pData[col][row];

			float real = val.real();
			float imag = val.imag();

			printf("(");
			if (real >= 0) printf("+");
			printf("%.3e ", real);
			if (imag >= 0) printf("+");
			else {
				imag = -imag;
				printf("-");
			}
			printf(" %.3ei)", imag);
			if (col != nCols - 1lu) printf(", ");
		}
		printf("\n");
	}
}

template<>
void vat::Mat<std::complex<double>>::print(std::string title) {

	if (title.length() > 0) std::cout << title << ":";
	 std::cout << std::endl;

	std::complex<double>** pData = subMat2D(0lu, 0lu, nRows - 1lu, nCols - 1lu);

	int leftBorderSize = floor(log10(nRows - 1lu)) + 1lu;
	int columnWidth = 27lu;

	printf("%*lu", leftBorderSize + 23, 0lu);
	for (ul col = 1; col < nCols; col++) {
		printf("%*lu", columnWidth, col);
	}   printf("\n\n");

	for (ul row = 0; row < nRows; row++) {

		printf("%*lu ", leftBorderSize, row);
		for (ul col = 0; col < nCols; col++) {

			std::complex<double> val;
			if (order == RowMajor) val = pData[row][col];
			else                   val = pData[col][row];

			double real = val.real();
			double imag = val.imag();

			printf("(");
			if (real >= 0) printf("+");
			printf("%.3e ", real);
			if (imag >= 0) printf("+");
			else {
				imag = -imag;
				printf("-");
			}
			printf(" %.3ei)", imag);
			if (col != nCols - 1lu) printf(", ");
		}
		printf("\n");
	}
}

template<typename T>
void vat::Mat<T>::printGrid() {

	grid->print();
}

template<typename T>
void vat::Mat<T>::matrixToGridIdx(ul& startRow, ul& startCol, ul& endRow, ul& endCol) {

	startRow += rowOffset;
	endRow   += rowOffset;
	startCol += colOffset;
	endCol   += colOffset;
}

template<typename T>
bool vat::Mat<T>::checkIndices(ul startRow, ul startCol, ul endRow, ul endCol) {

	bool validIndices = true;
	if (startRow >= nRows && endRow >= nRows && startRow > endRow) {
		validIndices = false;
		printf("Error: Matrix indices out of bounds\n");
	}

	if (startCol >= nCols && endCol >= nCols && startCol > endCol) {
		validIndices = false;
		printf("Error: Matrix indices out of bounds\n");
	}
	return validIndices;
}

template<typename T>
void vat::Mat<T>::setView(ul viewStartRow, ul viewStartCol, ul _nRows, ul _nCols) {

	// View matrix global grid bounds
	ul viewRowOffset = viewStartRow + rowOffset;
	ul viewColOffset = viewStartCol + colOffset;

	// Base matrix global grid bounds
	ul baseEndRow = nRows + rowOffset - 1lu;
	ul baseEndCol = nCols + colOffset - 1lu;

	// Candidate global offset
	ul viewEndRow = _nRows + viewRowOffset - 1lu;
	ul viewEndCol = _nCols + viewColOffset - 1lu;

	// Make sure the offset is in the bounds of the base view
	if (viewEndRow <= baseEndRow && viewEndCol <= baseEndCol) {

		rowOffset += viewRowOffset;
		colOffset += viewColOffset;

		nRows = _nRows;
		nCols = _nCols;
	} else {
		printf("Error: matrix sub-view bounds are out of bounds of the\n"
				"parent matrix\n");
	}
}

template<typename T>
void vat::Mat<T>::rotate(vat::RotationType rotationType, bool negate) {
    ul startRow = 0ul;
    ul startCol = 0ul;
    ul endRow = nRows - 1;
    ul endCol = nCols - 1;

    bool hasCenter = nRows % 2 != 0;

	ul layers   = hasCenter ? nRows / 2 : (nRows + 1) / 2;
	ul elements = (nRows - 1) * 4;

    if (nRows != nCols) throw std::logic_error("Error. Rotations only implemented for square matrices.");

	T* data = subMat(startRow, startCol, endRow, endCol);
    for (ul i = 0; i < layers; i++) {
        ul corner = i * nCols + i;
        ul batches = (ul)(elements / 4);

        for (ul j = 0; j < batches; j++) {
            const T ref1 = data[(int)(corner + j)];
            const T ref2 = data[(int)(corner + batches + (j * nCols))];

                const ul bottomCorner = corner + (batches * nCols);
                const T ref3 = data[(int)(bottomCorner - (j * nCols))];
                const T ref4 = data[(int)(bottomCorner + batches - j)];

                switch (rotationType) {
                    case RotationType::CCW:
                        data[(int)(corner + j)]                     = negate ? -ref2: ref2;
                        data[(int)(corner + batches + (j * nCols))] = negate ? -ref4: ref4;
                        data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref1: ref1;
                        data[(int)(bottomCorner + batches - j)]     = negate ? -ref3: ref3;
                        break;
                    case RotationType::CW:
                        data[(int)(corner + j)]                     = negate ? -ref3: ref3;
                        data[(int)(corner + batches + (j * nCols))] = negate ? -ref1: ref1;
                        data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref4: ref4;
                        data[(int)(bottomCorner + batches - j)]     = negate ? -ref2: ref2;
                        break;
                    case RotationType::Flip:
                        data[(int)(corner + j)]                     = negate ? -ref4: ref4;
                        data[(int)(corner + batches + (j * nCols))] = negate ? -ref3: ref3;
                        data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref2: ref2;
                        data[(int)(bottomCorner + batches - j)]     = negate ? -ref1: ref1;
                        break;
                }
            }
            //8 elements lost per layer increase
            elements -= 8;
        }
    subMat(data, startRow, startCol, endRow, endCol);
    if (data != nullptr) delete[] data;
}

template<typename T>
void vat::Mat<T>::rotate(vat::RotationType rotationType, vat::Mat<T>* mat, bool negate) {
    ul startRow = 0ul;
    ul startCol = 0ul;
    ul endRow = nRows - 1;
    ul endCol = nCols - 1;

    bool hasCenter = nRows % 2 != 0;

	ul layers   = hasCenter ? nRows / 2 : (nRows + 1) / 2;
	ul elements = (nRows - 1) * 4;

    if (nRows != nCols) throw std::logic_error("Error. Rotations only implemented for square matrices.");
    if (nRows != mat->getN_Rows() || nCols != mat->getN_Cols())
    	throw new std::runtime_error("Reference matrix must have the same dimensions as the parent matrix.");

	T* data = mat->subMat(startRow, startCol, endRow, endCol);
    for (ul i = 0; i < layers; i++) {
        ul corner = i * nCols + i;
        ul batches = (ul)(elements / 4);

        for (ul j = 0; j < batches; j++) {
            const T ref1 = data[(int)(corner + j)];
            const T ref2 = data[(int)(corner + batches + (j * nCols))];

                const ul bottomCorner = corner + (batches * nCols);
                const T ref3 = data[(int)(bottomCorner - (j * nCols))];
                const T ref4 = data[(int)(bottomCorner + batches - j)];

                switch (rotationType) {
                	case RotationType::CCW:
                		data[(int)(corner + j)]						= negate ? -ref2: ref2;
                        data[(int)(corner + batches + (j * nCols))] = negate ? -ref4: ref4;
                        data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref1: ref1;
                        data[(int)(bottomCorner + batches - j)]     = negate ? -ref3: ref3;
                        break;
                	case RotationType::CW:
                		data[(int)(corner + j)]                     = negate ? -ref3: ref3;
                		data[(int)(corner + batches + (j * nCols))] = negate ? -ref1: ref1;
                		data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref4: ref4;
                		data[(int)(bottomCorner + batches - j)]     = negate ? -ref2: ref2;
                		break;
                	case RotationType::Flip:
                		data[(int)(corner + j)]                     = negate ? -ref4: ref4;
                		data[(int)(corner + batches + (j * nCols))] = negate ? -ref3: ref3;
                		data[(int)(bottomCorner - (j * nCols))]     = negate ? -ref2: ref2;
                		data[(int)(bottomCorner + batches - j)]     = negate ? -ref1: ref1;
                		break;
                }
            }
            //8 elements lost per layer increase
            elements -= 8;
        }
    subMat(data, startRow, startCol, endRow, endCol);
    if (data != nullptr) delete[] data;
}

//special case of rotate
template<typename T>
void vat::Mat<T>::rotate(vat::Mat<T>* mat, bool negate) {
    for (int i = 0; i < nRows; i++) {
    	for (int j = 0; j < nCols; j++) {
    		const T elem = mat->element(nRows - 1 - i, nRows - 1 - j);
    		this->element(negate ? -elem : elem, i, j);
    	}
    }
}