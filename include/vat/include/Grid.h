/*
 * Grid.h
 *
 *  Created on: Sep 9, 2017
 *      Author: lance
 */

#ifndef GRID_H_
#define GIRD_H_

#include "Array.h"
#include <vector>

namespace vat {

template<typename T>
class Grid {
public:

	class Block;

	Grid(DiskCache* cache, GridDims _gridDims, BlockDims _blockDims, Order _order = RowMajor);
	virtual ~Grid();

	/*
	 * GETTERS
	 */

	BlockDims getBlockDims();
	GridDims  getGridDims();

	// Get rows in a 1D format
	T*  rows(ul startRow, ul endRow);

	// Get rows in a 2D format
	T** rows2D(ul startRow, ul endRow);

	// Get cols in a 1D format
	T*  cols(ul startCol, ul endCol);

	// Get cols in a 2D format
	T** cols2D(ul startCol, ul endCol);

	// Get a SubGrid in a 1D format
	T*  subGrid(ul startRow, ul startCol, ul endRow, ul endCol);

	// Get a SubGrid in a 1D format
	T** subGrid2D(ul startRow, ul startCol, ul endRow, ul endCol);

	/*
	 * SETTERS
	 */

	// Set rows with a 1D array
	void rows(T* pData, ul startRow, ul endRow);

	// Set rows with a 2D array
	void rows(T** pData, ul startRow, ul endRow);

	// Set cols with a 1D array
	void cols(T* pData, ul startCol, ul endCol);

	// Set cols with a 2D array
	void cols(T** pData, ul startCol, ul endCol);

	// Set a SubGrid with a 1D array
	void subGrid(T* pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// Set a SubGrid with a 2D array
	void subGrid(T** pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// Fill all blocks with fillVal
	void fill(T fillVal);

	void print();

private:

	GridDims  gridDims;
	BlockDims blockDims;
	GridDims  internalGridDims;
	BlockDims internalBlockDims;
	Order order;


	// Structure of blocks according to the internal order
	std::vector<std::vector<Block*>*>* blocks;
	
	bool checkGridIndexing(ul start, ul end, bool row);

	/*	
	 * GETTERS
	 */

	// Read from a subsection of the grid into a 1D array
	T*  internalSubGrid(ul startRow, ul startCol, ul endRow, ul endCol);

	// Read from a subsection of the grid into a 2D array
	T** internalSubGrid2D(ul startRow, ul startCol, ul endRow, ul endCol);

	/*
	 * SETTTERS
	 */
	
	// Write to a subsection of the grid with 1D array
	void internalSubGrid(T* pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// Write to a subsection of the grid with 2D array
	void internalSubGrid(T** pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// Change a 2D array to a 1D array
	static T* to1D(T** pData, ul nRows, ul nCols);

	// Change a 1D array to a 2D array of the same order
	static T** to2D(T* pData, ul nRows, ul nCols);
};

template class vat::Grid<float>;
template class vat::Grid<double>;
template class vat::Grid<std::complex<float>>;
template class vat::Grid<std::complex<double>>;

// A block is a component of a grid which is continuous in cache memory
template<typename T>
class Grid<T>::Block {
public:

	Block(DiskCache* cache, ul _nRows, ul _nCols, bool rowMajor = true);
	Block(DiskCache* cache, ul _nRows, ul _nCols, Order _order);
	~Block();

	/*
	 * GETTERS
	 */

	// Get row chunks in the specified format
	T*  rows(ul startRow, ul endRow);

	// Get row chunks in a 2D format of the block
	T** rows2D(ul startRow, ul endRow);

	// Get column chunks in a 1D format of the block
	T*  cols(ul startRow, ul endRow);

	// Get column chunk in a 2D format of the block
	T** cols2D(ul startCol, ul endCol);

	// Get subBlock in a 1D format of the block
	T*  subBlock(ul startRow, ul startCol, ul endRow, ul endCol);

	// Get subBlock in a 2D format of the block
	T** subBlock2D(ul startRow, ul startCol, ul endRow, ul endCol);

	/*
	 * SETTERS
	 */

	// Set row chunk with a 1D array in format of block
	void rows(T* pData, ul startRow, ul endRow);

	// Set row chunk with a 2D array in format of block
	void rows(T** pData, ul startRow, ul endRow);

	// Set column chunk with 1D array in format of the block
	void cols(T* pData, ul startCol, ul endCol);

	// Set column chunk with 2D array in format of the block
	void cols(T** pData, ul startCol, ul endCol);

	// set subBlock in a 1D format of the block
	void subBlock(T* pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// set subBlock in a 2D format of the block
	void subBlock(T** pData, ul startRow, ul startCol, ul endRow, ul endCol);

	void fill(T fillVal);

	void print();

private:

	ul nRows;
	ul nCols;
	ul nInternalRows;
	ul nInternalCols;
	ul nElements;
	Order order;
	Array<T>* data;

	void init(DiskCache* cache, ul _nRows, ul _nCols, Order order);

	// Validates that block indices are correct
	bool checkBlockIndexing(ul start, ul end, bool row);

	/*
	 * GETTERS
	 *
	 * Methods for retrieving continuous and strided blocks of data. Cache Rows
	 * are always continuous ranges while cache columns are not. There is a
	 * distinction because block rows may or may not be desired to be stored
	 * continuously
	 *
	 * Expensive to get strided blocks one by one, instead cache col accessors
	 * load all the data from the block into memory and then 'carve away' excess
	 * to get a result
	 *
	 * 1D and 2D return types give data in row major format
	 */

	// Get cache SubBlock in a 1D array
	T*  internalSubBlock(ul startRow, ul startCol, ul endRow, ul endCol);

	// Get cache SubBlock in a 2D array
	T** internalSubBlock2D(ul startRow, ul startCol, ul endRow, ul endCol);


	/*
	 * SETTERS
	 */

	// Set Cache SubBlock with 1D array
	void internalSubBlock(T* pData, ul startRow, ul startCol, ul endRow, ul endCol);

	// Set Cache subBlock with 2D array
	void internalSubBlock(T** pData, ul startRow, ul startCol, ul endRow, ul endCol);
};

template<> void Grid<float>::Block::fill(float fillVal);
template<> void Grid<double>::Block::fill(double fillVal);
template<> void Grid<std::complex<float>>::Block::fill(std::complex<float> fillVal);
template<> void Grid<std::complex<double>>::Block::fill(std::complex<double> fillVal);

} // End namespace



#endif /* GRID_H_ */
