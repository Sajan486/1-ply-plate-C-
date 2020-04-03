/*
 * Matrix.h
 *
 *  Created on: Sep 12, 2017
 *      Author: lance
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "Grid.h"
#include <math.h>
#include <iostream>
#include <string>

namespace vat {

template<typename T>
class Mat {
public:

	Mat(DiskCache* _cache, ul _nRows, ul _nCols, MemorySize maxBlockSize, Order _order = Order::RowMajor);
	Mat(const Mat<T>& mat);
	virtual ~Mat();

	/*
	 * ACCESSORS
	 */

	DiskCache* getCache();
	BlockDims getBlockDims();
	GridDims  getGridDims();

	T*  subMat  (ul startRow, ul startCol, ul endRow, ul endCol);
	T** subMat2D(ul startRow, ul startCol, ul endRow, ul endCol);
	T*  rows    (ul startRow, ul endRow);
	T** rows2D  (ul startRow, ul endRow);
	T*  row     (ul rowIdx);
	T*  cols    (ul startCol, ul endCol);
	T** cols2D  (ul startCol, ul endCol);
	T*  col     (ul colIdx);
	T   element (ul rowIdx, ul colIdx);
	T operator()(ul rowIdx, ul colIdx);

	/*
	 * SETTERS
	 */

	void subMat (T*  pData, ul startRow, ul startCol, ul endRow, ul endCol);
	void subMat (T** pData, ul startRow, ul startCol, ul endRow, ul endCol);
	void rows   (T*  pData, ul startRow, ul endRow);
	void rows2D (T** pData, ul startRow, ul endRow);
	void row    (T*  pData, ul rowIdx);
	void cols   (T*  pData, ul startCol, ul endCol);
	void cols2D (T** pData, ul startCol, ul endCol);
	void col    (T*  pData, ul colIdx);
	void element(T   val, ul rowIdx, ul colIdx);

	// Get a reference to a subsection of the Matrix
	Mat<T>* subView(ul startRow, ul startCol, ul endRow, ul endCol);

	void fill(T fillVal);

	ul getN_Rows();
	ul getN_Cols();

	void print(std::string title = "");
	void printGrid();
	
	void rotate(vat::RotationType rotationType, bool negate = false);
	void rotate(vat::RotationType rotationType, vat::Mat<T>* mat, bool negate = false);
	void rotate(vat::Mat<T>* mat, bool negate = false);

private:

	ul nRows;
	ul nCols;
	Order order;

	// For defining the size of a sub-view
	ul rowOffset;
	ul colOffset;

	// Matrix data
	Grid<T>* grid;
	DiskCache* cache;

	// Is this the base view?
	bool base;

	// Use offsets to convert to the view indices
	void matrixToGridIdx(ul& startRow, ul& startCol, ul& endRow, ul& endCol);

	// Check that the indices are within the bounds of the view
	bool checkIndices(ul startRow, ul startCol, ul endRow, ul endCol);

	/*
	 * Set offsets for the view, using this as alternative to new constructor.
	 * Don't want to expose this
	 */
	void setView(ul viewStartRow, ul viewStartCol, ul _nRows, ul _nCols);
};

template class Mat<float>;
template class Mat<double>;
template class Mat<std::complex<float>>;
template class Mat<std::complex<double>>;

template<> void Mat<float>::print(std::string title);
template<> void Mat<double>::print(std::string title);
template<> void Mat<std::complex<float>>::print(std::string title);
template<> void Mat<std::complex<double>>::print(std::string title);

template<typename F>
using CxMat = Mat<std::complex<F>>;

typedef Mat<std::complex<float>>  CxfMat ;
typedef Mat<std::complex<double>> CxdMat;

} // End namespace

#endif /* MATRIX_H_ */
