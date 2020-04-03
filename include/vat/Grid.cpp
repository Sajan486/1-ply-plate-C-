/*
 * Grid.cpp
 *
 *  Created on: Sep 9, 2017
 *      Author: lance
 */

#include "Grid.h"

template<typename T>
vat::Grid<T>::Grid(DiskCache* cache, GridDims _gridDims, BlockDims _blockDims, Order _order) {

	order = _order;
	gridDims  = _gridDims;
	blockDims = _blockDims;
	internalGridDims  = gridDims;
	internalBlockDims = blockDims;

	if (order != RowMajor) {
		internalGridDims.x  = internalGridDims.y;
		internalGridDims.y  = internalGridDims.x;
		internalBlockDims.x = internalBlockDims.y;
		internalBlockDims.y = internalBlockDims.x;
	}

	blocks = new std::vector<std::vector<Block*>*>;

	for (ul row = 0lu; row < internalGridDims.y; row++) {
		std::vector<Block*>* rowPtr = new std::vector<Block*>;
		for (ul col = 0lu; col < internalGridDims.x; col++) {
			rowPtr->push_back(new Block(cache, internalBlockDims.y, internalBlockDims.x, RowMajor));
		}
		blocks->push_back(rowPtr);
	}
}

template<typename T>
vat::Grid<T>::~Grid() {
	for (ul row = 0lu; row < internalGridDims.y; row++) {
		std::vector<Block*>* rowPtr = blocks->at(row);
		for (ul col = 0lu; col < internalGridDims.x; col++) {
			Block* block = rowPtr->at(col);
			delete block;
		}
		delete rowPtr;
	}
	delete blocks;
}

template<typename T>
void vat::Grid<T>::fill(T fillVal) {

	for (ul row = 0lu; row < internalGridDims.y; row++) {
		std::vector<Block*>* rowPtr = blocks->at(row);
		for (ul col = 0lu; col < internalGridDims.x; col++) {
			Block* block = rowPtr->at(col);
			block->fill(fillVal);
		}
	}
}

/*
 * GETTERS
 */

template<typename T>
vat::GridDims vat::Grid<T>::getGridDims() {
	return gridDims;
}

template<typename T>
vat::BlockDims vat::Grid<T>::getBlockDims() {
	return blockDims;
}

template<typename T>
T* vat::Grid<T>::rows(ul startRow, ul endRow) {

	ul nCols = gridDims.x * blockDims.x;
	return subGrid(startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
T** vat::Grid<T>::rows2D(ul startRow, ul endRow) {

	ul nCols = gridDims.x * blockDims.x;
	return subGrid2D(startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
T* vat::Grid<T>::cols(ul startCol, ul endCol) {

	ul nRows = gridDims.y * blockDims.y;
	return subGrid(0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
T** vat::Grid<T>::cols2D(ul startCol, ul endCol) {

	ul nRows = gridDims.y * blockDims.y;
	return subGrid2D(0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
T* vat::Grid<T>::subGrid(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubGrid(startRow, startCol, endRow, endCol);
	else                   return internalSubGrid(startCol, startRow, endCol, endRow);
}

template<typename T>
T** vat::Grid<T>::subGrid2D(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubGrid2D(startRow, startCol, endRow, endCol);
	else                   return internalSubGrid2D(startCol, startRow, endCol, endRow);
}

/*
 * SETTERS
 */

template<typename T>
void vat::Grid<T>::rows(T* pData, ul startRow, ul endRow) {

	ul nCols = gridDims.x * blockDims.x;
	subGrid(pData, startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
void vat::Grid<T>::rows(T** pData, ul startRow, ul endRow) {

	ul nCols = gridDims.x * blockDims.x;
	subGrid(pData, startRow, 0lu, endRow, nCols - 1lu);
}

template<typename T>
void vat::Grid<T>::cols(T* pData, ul startCol, ul endCol) {

	ul nRows = gridDims.y * blockDims.y;
	subGrid(pData, 0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
void vat::Grid<T>::cols(T** pData, ul startCol, ul endCol) {

	ul nRows = gridDims.y * blockDims.y;
	subGrid(pData, 0lu, startCol, nRows - 1lu, endCol);
}

template<typename T>
void vat::Grid<T>::subGrid(T* pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubGrid(pData, startRow, startCol, endRow, endCol);
	else                   return internalSubGrid(pData, startCol, startRow, endCol, endRow);
}

template<typename T>
void vat::Grid<T>::subGrid(T** pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubGrid(pData, startRow, startCol, endRow, endCol);
	else                   return internalSubGrid(pData, startCol, startRow, endCol, endRow);
}

/*
 * Very inefficient, for debugging really
 * Only does floats and doubles
 */
template<typename T>
void vat::Grid<T>::print() {

	ul nRows = gridDims.y * blockDims.y;
	ul nCols = gridDims.x * blockDims.x;

	T* pData = subGrid(0lu, 0lu, nRows - 1lu, nCols - 1lu);
	for (ul row = 0lu; row < nRows; row++) {
		if (row % blockDims.y == 0) printf("\n\n");
		for (ul col = 0lu; col < nCols; col++) {
			T val;
			if (order == RowMajor) val = pData[col + nCols * row];
			else                   val = pData[row + nRows * col];

			if (col % blockDims.x == 0) printf("   ");
			printf("%.2lf", val);
			if (col != nCols - 1lu) printf(", ");
		}
		printf("\n");
	}
}

template<typename T>
bool vat::Grid<T>::checkGridIndexing(ul start, ul end, bool rows) {

	ul nRows = internalGridDims.y * internalBlockDims.y;
	ul nCols = internalGridDims.x * internalBlockDims.x;

	if (rows) {
		if ((start < nRows) && (end < nRows) && (start <= end)) return true;
		else {
			printf("Error: Grid indexing out of bounds\n");
		}
	} else {
		if ((start < nCols) && (end < nCols) && (start <= end)) return true;
		else {
			printf("Error: Grid indexing out of bounds\n");
		}
	}
}

/*
 * GETTERS
 */

template<typename T>
T* vat::Grid<T>::internalSubGrid(ul gStartRow, ul gStartCol, ul gEndRow, ul gEndCol) {


	if (checkGridIndexing(gStartRow, gEndRow, true) && checkGridIndexing(gStartCol, gEndCol, false)) {

		// Get the global grid bounds
		ul startGridRow = gStartRow / internalBlockDims.y;
		ul startGridCol = gStartCol / internalBlockDims.x;
		ul endGridRow   = gEndRow   / internalBlockDims.y;
		ul endGridCol   = gEndCol   / internalBlockDims.x;
		ul g_nRows      = gEndRow - gStartRow + 1lu;
		ul g_nCols      = gEndCol - gStartCol + 1lu;

		T* pData = new T[g_nRows * g_nCols];

		for (ul gridRow = startGridRow; gridRow <= endGridRow; gridRow++) {
			for (ul gridCol = startGridCol; gridCol <= endGridCol; gridCol++) {

				Block* block = blocks->at(gridRow)->at(gridCol);

				// Get the global element bounds of the current block
				ul lStartRow = gridRow * internalBlockDims.y;
				ul lStartCol = gridCol * internalBlockDims.x;
				ul lEndRow   = lStartRow + internalBlockDims.y - 1lu;
				ul lEndCol   = lStartCol + internalBlockDims.x - 1lu;

				// Truncate block bounds based on the global bounds
				if (lStartRow < gStartRow) lStartRow = gStartRow;
				if (lStartCol < gStartCol) lStartCol = gStartCol;
				if (lEndRow   > gEndRow)   lEndRow   = gEndRow;
				if (lEndCol   > gEndCol)   lEndCol   = gEndCol;

				ul pDataRowOffset = lStartRow - gStartRow;
				ul pDataColOffset = lStartCol - gStartCol;

				// Transform global element bounds to local truncated block bounds
				lStartRow -= gridRow * internalBlockDims.y;
				lStartCol -= gridCol * internalBlockDims.x;
				lEndRow   -= gridRow * internalBlockDims.y;
				lEndCol   -= gridCol * internalBlockDims.x;
				ul nRows  = lEndRow - lStartRow + 1lu;
				ul nCols  = lEndCol - lStartCol + 1lu;

				T** bData = block->subBlock2D(lStartRow, lStartCol, lEndRow, lEndCol);

				// Copy each piece of data into the return val
				for (ul bRow = 0lu; bRow < nRows; bRow++) {

					T* destPos = pData + pDataColOffset + (bRow + pDataRowOffset) * g_nCols;
					memcpy(destPos, bData[bRow], nCols * sizeof(T));
				}
			}
		}
		return pData;
	}
	else return (T*)(0);
}

template<typename T>
T** vat::Grid<T>::internalSubGrid2D(ul gStartRow, ul gStartCol, ul gEndRow, ul gEndCol) {
	ul nRows = gEndRow - gStartRow + 1lu;
	ul nCols = gEndCol - gStartCol + 1lu;

	T* pData1D = internalSubGrid(gStartRow, gStartCol, gEndRow, gEndCol);
	return to2D(pData1D, nRows, nCols);
}


/*
 * SETTERS
 */

template<typename T>
void vat::Grid<T>::internalSubGrid(T* pData, ul gStartRow, ul gStartCol, ul gEndRow, ul gEndCol) {
	ul nRows = gEndRow - gStartRow + 1lu;
	ul nCols = gEndCol - gStartCol + 1lu;

	T** pData2D = to2D(pData, nRows, nCols);

	internalSubGrid(pData2D, gStartRow, gStartCol, gEndRow, gEndCol);
	delete[] pData2D;
}

template<typename T>
void vat::Grid<T>::internalSubGrid(T** pData, ul gStartRow, ul gStartCol, ul gEndRow, ul gEndCol) {

	if (checkGridIndexing(gStartRow, gEndRow, true) && checkGridIndexing(gStartCol, gEndCol, false)) {
		// Get the global grid bounds
		ul startGridRow = gStartRow / internalBlockDims.y;
		ul startGridCol = gStartCol / internalBlockDims.x;
		ul endGridRow   = gEndRow   / internalBlockDims.y;
		ul endGridCol   = gEndCol   / internalBlockDims.x;

		for (ul gridRow = startGridRow; gridRow <= endGridRow; gridRow++) {
			for (ul gridCol = startGridCol; gridCol <= endGridCol; gridCol++) {

				Block* block = blocks->at(gridRow)->at(gridCol);

				// Get the global element bounds of the current block
				ul lStartRow = gridRow * internalBlockDims.y;
				ul lStartCol = gridCol * internalBlockDims.x;
				ul lEndRow   = lStartRow + internalBlockDims.y - 1lu;
				ul lEndCol   = lStartCol + internalBlockDims.x - 1lu;

				// Truncate block bounds based on the global bounds
				if (lStartRow < gStartRow) lStartRow = gStartRow;
				if (lStartCol < gStartCol) lStartCol = gStartCol;
				if (lEndRow   > gEndRow)   lEndRow   = gEndRow;
				if (lEndCol   > gEndCol)   lEndCol   = gEndCol;

				ul pDataRowOffset = lStartRow - gStartRow;
				ul pDataColOffset = lStartCol - gStartCol;

				// Transform global element bounds to local truncated block bounds
				lStartRow -= gridRow * internalBlockDims.y;
				lStartCol -= gridCol * internalBlockDims.x;
				lEndRow   -= gridRow * internalBlockDims.y;
				lEndCol   -= gridCol * internalBlockDims.x;
				ul nRows  = lEndRow - lStartRow + 1lu;
				ul nCols  = lEndCol - lStartCol + 1lu;

				T** bData = new T*[nRows];
				for (ul row = 0; row < nRows; row++) {
					bData[row] = pData[row + pDataRowOffset] + pDataColOffset;
				}

				block->subBlock(bData, lStartRow, lStartCol, lEndRow, lEndCol);
				delete[] bData;
			}
		}
	}
}

template<typename T>
T* vat::Grid<T>::to1D(T** pData, ul _nRows, ul _nCols) {
	T* oData = new T[_nRows * _nCols];
	for (ul i = 0lu; i < _nRows; i++) {
		for (ul j = 0lu; j < _nCols; j++) {
			oData[j + _nCols * i] = pData[i][j];
		}
	}

	return oData;
}

template<typename T>
T** vat::Grid<T>::to2D(T* pData, ul _nRows, ul _nCols) {

	T** oData = new T*[_nRows];
	for (ul i = 0lu; i < _nRows; i++) {
		oData[i] = pData + i * _nCols;
	}
	return oData;
}

/*
 * Grid<T>::Block definitions
 */

template<typename T>
vat::Grid<T>::Block::Block(DiskCache* cache, ul _nRows, ul _nCols, bool rowMajor) {
	Order _pattern;
	if (rowMajor) _pattern = RowMajor;
	else _pattern = ColumnMajor;
	init(cache, _nRows, _nCols, _pattern);
}

template<typename T>
vat::Grid<T>::Block::Block(DiskCache* cache, ul _nRows, ul _nCols, Order _order) {
	init(cache, _nRows, _nCols, _order);
}

template<typename T>
void vat::Grid<T>::Block::init(DiskCache* cache, ul _nRows, ul _nCols, Order _order) {
	nRows = _nRows;
	nCols = _nCols;
	nElements = nRows * nCols;
	data = new Array<T>(cache, nElements);
	order = _order;
	if (order == RowMajor) {
		nInternalRows = nRows;
		nInternalCols = nCols;
	} else {
		nInternalRows = nCols;
		nInternalCols = nRows;
	}
}

template<typename T>
vat::Grid<T>::Block::~Block() {
	delete data;
}

template<typename T>
bool vat::Grid<T>::Block::checkBlockIndexing(ul start, ul end, bool rows) {

	if (rows) {
		if ((start < nInternalRows) && (end < nInternalRows) && (end >= start)) return true;
		else {
			printf("Error: Grid::Block indexing out of bounds\n");
		}
	} else  {
		if ((start < nInternalCols) && (end < nInternalCols) && (end >= start)) return true;
		else {
			printf("Error: Grid::Block indexing out of bounds\n");
		}
	}
}

/*
 * GETTERS
 */

template<typename T>
T* vat::Grid<T>::Block::rows(ul startRow, ul endRow) {

	if (order == RowMajor) return internalSubBlock(startRow, 0lu, endRow, nInternalCols - 1lu);
	else                   return internalSubBlock(0lu, startRow, nInternalCols - 1lu, endRow);
}

template<typename T>
T** vat::Grid<T>::Block::rows2D(ul startRow, ul endRow) {

	if (order == RowMajor) return internalSubBlock2D(startRow, 0lu, endRow, nInternalCols - 1lu);
	else                   return internalSubBlock2D(0lu, startRow, nInternalCols - 1lu, endRow);
}

template<typename T>
T* vat::Grid<T>::Block::cols(ul startCol, ul endCol) {

	if (order == RowMajor) return internalSubBlock(0lu, startCol, nInternalRows - 1lu, endCol);
	else                   return internalSubBlock(startCol, 0lu, endCol, nInternalRows - 1lu);
}

template<typename T>
T** vat::Grid<T>::Block::cols2D(ul startRow, ul endRow) {

	if (order == RowMajor) return internalSubBlock2D(0lu, startRow, nInternalCols - 1lu, endRow);
	else                   return internalSubBlock2D(startRow, 0lu, endRow, nInternalCols - 1lu);
}

template<typename T>
T** vat::Grid<T>::Block::internalSubBlock2D(ul startRow, ul startCol, ul endRow, ul endCol) {
	ul _nRows = endRow - startRow + 1lu;
	ul _nCols = endCol - startCol + 1lu;

	T* pData1D = internalSubBlock(startRow, startCol, endRow, endCol);
	T** pData2D = to2D(pData1D, _nRows, _nCols);
	return pData2D;
}

template<typename T>
T* vat::Grid<T>::Block::subBlock(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubBlock(startRow, startCol, endRow, endCol);
	else                   return internalSubBlock(startCol, startRow, endCol, endRow);
}

template<typename T>
T** vat::Grid<T>::Block::subBlock2D(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) return internalSubBlock2D(startRow, startCol, endRow, endCol);
	else                   return internalSubBlock2D(startCol, startRow, endCol, endRow);
}

/*
 * SETTERS
 */

template<typename T>
void vat::Grid<T>::Block::rows(T* pData, ul startRow, ul endRow) {

	if (order == RowMajor) internalSubBlock(pData, startRow, 0lu, endRow, nInternalCols - 1lu);
	else                   internalSubBlock(pData, 0lu, startRow, nInternalCols - 1lu, endRow);
}

template<typename T>
void vat::Grid<T>::Block::rows(T** pData, ul startRow, ul endRow) {

	if (order == RowMajor) internalSubBlock(pData, startRow, 0lu, endRow, nInternalCols - 1lu);
	else                   internalSubBlock(pData, 0lu, startRow, nInternalCols - 1lu, endRow);
}

template<typename T>
void vat::Grid<T>::Block::cols(T* pData, ul startRow, ul endRow) {

	if (order == RowMajor) internalSubBlock(pData, 0lu, startRow, nInternalCols - 1lu, endRow);
	else                   internalSubBlock(pData, startRow, 0lu, endRow, nInternalCols - 1lu);
}

template<typename T>
void vat::Grid<T>::Block::cols(T** pData, ul startRow, ul endRow) {

	if (order == RowMajor) internalSubBlock(pData, 0lu, startRow, nInternalCols - 1lu, endRow);
	else                   internalSubBlock(pData, startRow, 0lu, endRow, nInternalCols - 1lu);
}

template<typename T>
void vat::Grid<T>::Block::subBlock(T* pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) internalSubBlock(pData, startRow, startCol, endRow, endCol);
	else                   internalSubBlock(pData, startCol, startRow, endCol, endRow);
}

template<typename T>
void vat::Grid<T>::Block::subBlock(T** pData, ul startRow, ul startCol, ul endRow, ul endCol) {

	if (order == RowMajor) internalSubBlock(pData, startRow, startCol, endRow, endCol);
	else                   internalSubBlock(pData, startCol, startRow, endCol, endRow);
}

template<typename T>
void vat::Grid<T>::Block::print() {
	T** data = rows2D(0, nInternalRows - 1lu);

	for (ul i = 0lu; i < nRows; i++) {
		for (ul j = 0lu; j < nCols; j++) {

			if (order == RowMajor) {
				printf("%.2lf", data[i][j]);
				if (j != nInternalCols - 1lu) printf(", ");
			} else {
				printf("%.2lf", data[j][i]);
				if (j != nInternalRows - 1lu) printf(", ");
			}
		}
		printf("\n");
	}
}

template<>
void vat::Grid<float>::Block::fill(float fillVal) {
	float* pData = new float[nElements];
	for (ul i = 0lu; i < nElements; i++) {
		pData[i] = fillVal;
	}
	data->write(pData, 0lu, nElements - 1lu);
	delete[] pData;
}

template<>
void vat::Grid<double>::Block::fill(double fillVal) {
	double* pData = new double[nElements];
	for (ul i = 0lu; i < nElements; i++) {
		pData[i] = fillVal;
	}
	data->write(pData, 0lu, nElements - 1lu);
	delete[] pData;
}

template<>
void vat::Grid<std::complex<float>>::Block::fill(std::complex<float> fillVal) {
	std::complex<float>* pData = new std::complex<float>[nElements];
	for (ul i = 0lu; i < nElements; i++) {
		pData[i] = fillVal;
	}
	data->write(pData, 0lu, nElements - 1lu);
	delete[] pData;
}

template<>
void vat::Grid<std::complex<double>>::Block::fill(std::complex<double> fillVal) {
	std::complex<double>* pData = new std::complex<double>[nElements];
	for (ul i = 0lu; i < nElements; i++) {
		pData[i] = fillVal;
	}
	data->write(pData, 0lu, nElements - 1lu);
	delete[] pData;
}

/*
 * GETTERS
 */

template<typename T>
T* vat::Grid<T>::Block::internalSubBlock(ul startRow, ul startCol, ul endRow, ul endCol) {

	if (checkBlockIndexing(startRow, endRow, true) && checkBlockIndexing(startCol, endCol, false)) {

		/*
		 * Pull data bounded by upper and lower rows
		 */
		ul _nRows = endRow - startRow + 1lu;
		ul _nCols = endCol - startCol + 1lu;
		ul start  = nInternalCols * startRow;
		ul end    = start + nInternalCols * _nRows - 1lu;

		T* buffer = data->read(start, end);

		// Test for an efficient read condition (is the data contiguous?)
		if (_nCols == nInternalCols || nRows == 1lu) {
			return buffer;
		}
		else {
			/*
			 * If the buffer isn't exactly what we need, we 'carve' it out of
			 * the buffer and put it into a new array
			 */
			T* pData = new T[_nRows * _nCols];
			for (ul row = 0lu; row < _nRows; row++) {
				for (ul col = 0lu; col < _nCols; col++) {
					pData[col + _nCols * row] = buffer[col + startCol + nInternalCols * row];
				}
			}
			delete[] buffer;
			return pData;
		}
	}
	else return (T*)(0);
}

/*
 * SETTERS
 */

template<typename T>
void vat::Grid<T>::Block::internalSubBlock(T* pData, ul startRow, ul startCol, ul endRow, ul endCol) {
	if (checkBlockIndexing(startRow, endRow, true) && checkBlockIndexing(startCol, endCol, false)) {

		ul _nRows = endRow - startRow + 1lu;
		ul _nCols = endCol - startCol + 1lu;

		// Array index bounds for rows
		ul start = nInternalCols * startRow;
		ul end   = start + nInternalCols * _nRows - 1lu;

		// Test for an efficient write condition (is the data contiguous?)
		if (_nCols == nInternalCols || nRows == 1lu) {

			data->write(pData, start, end);
		} else {

			// Read in a buffer
			T* buffer;
			buffer = data->read(start, end);

			for (ul row = 0lu; row < _nRows; row++) {
				for (ul col = 0lu; col < _nCols; col++) {
					buffer[col + startCol + nInternalCols * row] = pData[col + _nCols * row];
				}
			}
			data->write(buffer, start, end);
			delete[] buffer;
		}
	}
}

template<typename T>
void vat::Grid<T>::Block::internalSubBlock(T** pData, ul startRow, ul startCol, ul endRow, ul endCol) {
	ul _nRows = endRow - startRow + 1lu;
	ul _nCols = endCol - startCol + 1lu;

	T* pData1D = to1D(pData, _nRows, _nCols);

	internalSubBlock(pData1D, startRow, startCol, endRow, endCol);
	delete[] pData1D;
}


