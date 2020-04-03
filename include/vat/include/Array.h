#ifndef ARRAY_H_
#define ARRAY_H_

#include "DiskCache.h"

namespace vat {

template<typename T>
class Array {
public:
	Array(DiskCache* _cache, ul _targetSize);
	virtual ~Array();

	/*
	 * Re-allocate space for the array
	 * Note: this will allocate space in the nearest available block so data
	 * is not required to be preserved here
	 */
	void allocate(ul _size);

	/*
	 * TODO: Move fstream instances here so that read and write can be
	 * parallelized
	 */

	// Writes in core memory to parent cache
	void write(T* pData, ul startIdx, ul endIdx);

	// Reads continuous section of memory from parent cache.
	T* read(ul startIdx, ul endIdx);

	// Reads continuous section of memory into provided array
	T* read(T* pData, ul startIdx, ul endIdx);

	/*
	 * Reads regular sized pieces of cache memory separated by regular gaps into
	 * a continuous array
	 */
	T* read(ul offset, ul blockSize, ul gapSize);

	// Frees the memory block association with the parent cache
	void free();

private:
	DiskCache* cache;
	MemoryBlock* block;
	ul elementSize;

	// Read function generalized for allocation strategies
	T* read(T* pData, ul startIdx, ul endIdx, bool autoMalloc);

};

template class Array<float>;
template class Array<double>;
template class Array<std::complex<float>>;
template class Array<std::complex<double>>;

} // End namespace


#endif /* ARRAY_H_ */
