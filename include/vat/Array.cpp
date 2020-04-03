/*
 * Array.cpp
 *
 *  Created on: Sep 6, 2017
 *      Author: lance
 */

#include "Array.h"

template<typename T>
vat::Array<T>::Array(DiskCache* _cache, ul _targetSize) {
	cache = _cache;
	elementSize = sizeof(T);
	block = new MemoryBlock;
	block->start = 0lu;
	block->allocated = false;
	block->cache = (void*)cache;
	allocate(_targetSize);
}

template<typename T>
vat::Array<T>::~Array() {
	if (block->allocated) free();
	delete block;
}

template<typename T>
void vat::Array<T>::free() {
	cache->freeBlock(block);
}

template<typename T>
void vat::Array<T>::allocate(ul _targetSize) {
	if (block->allocated) free();
	block->size = elementSize * _targetSize;
	cache->allocateBlock(block);
}

template<typename T>
void vat::Array<T>::write(T* pData, ul startIdx, ul endIdx) {

	if (!block->allocated) {
		printf("Error: cannot write to unallocated block\n");
		return;
	}
	if (block->size == 0) {
		printf("Error: cannot write to an array with size zero\n");
		return;
	}

	ul start = elementSize * startIdx;
	ul end   = elementSize * endIdx;
	if (start > block->size - 1lu || endIdx > block->size - 1lu) {
		printf("Error: array index out of bounds in writing\n");
		return;
	}
	ul writeSize = endIdx - startIdx + 1lu;

	cache->writeData<T>(block->start + start, pData, writeSize);
}

template<typename T>
T* vat::Array<T>::read(ul startIdx, ul endIdx) {
	return read((T*)(0), startIdx, endIdx, true);
}

template<typename T>
T* vat::Array<T>::read(T* pData, ul startIdx, ul endIdx) {
	return read(pData, startIdx, endIdx, false);
}

template<typename T>
T* vat::Array<T>::read(T* pData, ul startIdx, ul endIdx, bool autoMalloc) {
	if (!block->allocated) {
		printf("Error: cannot read from unallocated block\n");
		return new T;
	}
	if (block->size == 0lu) {
		printf("Error: cannot read from an array with size zero\n");
		return new T;
	}
	ul start = elementSize * startIdx;
	ul end   = elementSize * endIdx;
	if (start > block->size - 1lu || end > block->size - 1lu || start > end) {
		printf("Error: array index out of bounds in reading\n");
		return new T;
	}
	ul readSize = endIdx - startIdx + 1lu;

	if (autoMalloc) return cache->readData<T>(block->start + start, readSize);
	else {
		cache->readData<T>(block->start + start, pData, readSize);
		return pData;
	}
}


