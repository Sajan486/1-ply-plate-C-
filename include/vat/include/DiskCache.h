/*
 * DiskCache.h
 *
 *  Created on: Sep 6, 2017
 *      Author: lance
 */

#ifndef DISKCACHE_H_
#define DISKCACHE_H_

#include "VatTypes.h"

#include <string>
#include <vector>
#include <fstream>
#include <unistd.h> // Retrieving file descriptor
#include <cstring>  // c error handling
#include <fcntl.h>  // posix_fallocate
#include <mutex>    // For preventing race conditions in IO

namespace vat {

/*
 * Basic disk cache
 *    + Allocates static space for a cache in a specified file
 *    + Allocates static space for arrays of basic data types within cache
 *    + Manages read/write of arrays
 *    + Frees space when arrays are removed
 *    + Frees cache on command or when the program ends
 */
class DiskCache {
public:

	template<typename T>
	friend class Array;

	/*
	 * Create a cache file with a specified size.
	 *
	 * If the cache already exists and is larger or smaller than the specified
	 * size, then the cache will be truncated to the specified size.
	 *
	 * If the specified size is larger than the amount of available memory on
	 * the device, then an exception will be thrown and no cache will be created
	 *
	 */
	DiskCache(std::string _path, MemorySize size, MemorySize _maxIO, bool _persistent = false);
	virtual ~DiskCache();

	/*
	 * Use an already existing file as a cache. This does not check to make sure
	 * that the cache is of a reasonable size, but will throw an exception if
	 * the file does not exist
	 */
	DiskCache(std::string _path, MemorySize _maxIO, bool _persistent = false);

	MemorySize getMaxIO();

	vat::MemorySize getFreeSize();
	double getFreeSize(vat::MemoryUnit unit);


private:

	// Constructor variables
	std::string path;          // path to cache file
	ul          cacheSize;     // in bytes
	ul          freeSize;      // how much cache is occupied?
	bool        persistent;    // does the cache file remain on object deletion?
	MemorySize  maxIO;         // Max input/output allowed

	// for IO
	std::fstream* cache;

	// Array of memory blocks ordered by their starting position in the cache
	std::vector<MemoryBlock*> blocks;

	// Mutex for preventing race conditions in parallel IOss
	static std::mutex mtx;

	// Handles the bulk of logic for both constructors
	void init(bool useExistingFile);

	/*
	 * Find free space for a block with a certain size.
	 * If there is not enough space in the cache or the block is already
	 * allocated then the block will not be allocated
	 */
	void allocateBlock(MemoryBlock* block);

	// If the block is currently allocated in the cache, then it is removed
	void freeBlock(MemoryBlock* block);

	// Writes an array of basic types into a file
	template<typename T>
	void writeData(ul filePos, T* pData, ul nElements) {

		ul nBytes = sizeof(T) * nElements;

		mtx.lock();
		cache->seekp(filePos, cache->beg);
		cache->write(reinterpret_cast<char*>(pData), std::streamsize(nBytes));
		mtx.unlock();
	}

	template<typename T>
	T* readData(ul filePos, ul nElements) {
		T* pData = new T[nElements];
		readData<T>(filePos, pData, nElements);
		return pData;
	}

	// Reads a continuous array of basic types out of a file
	template<typename T>
	void readData(ul filePos, T* pData, ul nElements) {

		ul nBytes = sizeof(T) * nElements;

		mtx.lock();
		cache->seekp(filePos, cache->beg);
		cache->read(reinterpret_cast<char*>(pData), std::streamsize(nBytes));
		mtx.unlock();
	}

	void clearCache();
};

}

#endif /* DISKCACHE_H_ */
