/*
 * DiskCache.cpp
 *
 *  Created on: Sep 6, 2017
 *      Author: lance
 */

#include "DiskCache.h"

std::mutex vat::DiskCache::mtx;

vat::DiskCache::DiskCache(std::string _path, MemorySize size, MemorySize _maxIO, bool _persistent) {
	path = _path;
	cacheSize = size.nBytes();
	maxIO = _maxIO;
	persistent = _persistent;

	init(false);
}

vat::DiskCache::DiskCache(std::string _path, MemorySize _maxIO, bool _persistent) {
	path = _path;
	maxIO = _maxIO;
	persistent = _persistent;

	init(true);
}

vat::MemorySize vat::DiskCache::getMaxIO() {
	return maxIO;
}

void vat::DiskCache::init(bool useExistingFile) {

	if (useExistingFile) {
		bool fileExists = (access(path.c_str(), F_OK) != -1);
		if (!fileExists) {
			char* msg = new char[1000];
			sprintf(msg, "%s", "Error: pre-allocated cache file does not exist");
			throw msg;
		}
	}

	/*
	 * Get the file descriptor for the specified path. If the file does not
	 * exist then create it. File is given read and write permission
	 *
	 * http://bit.ly/2gODhsH
	 * http://bit.ly/2gOo6Ql
	 */
	int fd = open(path.c_str(), O_RDWR | O_CREAT, S_IWUSR | S_IRUSR, 0777);

	// Check to see if the file open failed
	if (fd == -1) {
		char* msg = new char[1000];
		sprintf(msg, "Error: can't to open cache file: %s", strerror(errno));
		throw msg;
	}

	ul initSize = lseek(fd, 0L, SEEK_END);

	if (useExistingFile) cacheSize = initSize;

	/*
	 * cache size will always be known at this point. Therefore; we define the
	 * free size as the size of the new cache.
	 */
	freeSize = cacheSize;

	/*
	 * Allocates space in a file for the number of specified bytes. If the
	 * requested size is too large then it will allocate the maximum available
	 * space. If the file is larger than the number of specified bytes, then
	 * the file is truncated. File must be open to allocate
	 *
	 * http://bit.ly/2wJm1sN
	 */
	if (initSize < cacheSize) {
		errno = posix_fallocate(fd, 0, cacheSize);
	}
	else if (initSize > cacheSize) {
		errno = ftruncate(fd, cacheSize);
	}

	// Check that the cache was allocated properly
	if (errno != 0) {
		char* msg = new char[1000];
		sprintf(msg, "Error: can't set file size correctly: %s", strerror(errno));
		clearCache();
		throw msg;
	}

	// Check that file closed properly
	if (close(fd) < 0) {
		char* msg = new char[1000];
		sprintf(msg, "Error: cache file not closed: %s", strerror(errno));
		clearCache();
		throw msg;
	}

	cache = new std::fstream;
	cache->open(path, std::fstream::in | std::fstream::out | std::fstream::binary);

	/*
	 * Define 'dummy' end block for finding occupancies
	 */
	MemoryBlock* endBlock = new MemoryBlock;
	endBlock->start = cacheSize;
	endBlock->size  = 0;
	blocks.push_back(endBlock);
}

/*
 * Cache is not meant to be persistent (unless specified), so we need to delete
 * it when the cache is no longer available in the program
 */
vat::DiskCache::~DiskCache() {

	cache->close();
	delete cache;

	if (!persistent) clearCache();

	delete blocks.at(blocks.size() - 1);
}

vat::MemorySize vat::DiskCache::getFreeSize() {
	return vat::MemorySize(freeSize, vat::B);
}

double vat::DiskCache::getFreeSize(vat::MemoryUnit unit) {
	return vat::MemorySize(freeSize, vat::B).sizeIn(unit);
}

void vat::DiskCache::allocateBlock(MemoryBlock* block) {

	// Check that there is enough space in the cache to hold the array
	if (freeSize >= block->size && !block->allocated) {

		/*
		 * Look through cache vacancies for the first location that would
		 * provide continuous storage
		 */
		bool vacancyFound = false;
		ul prevEnd = -1;
		for (ul i = 0; i < blocks.size(); i++) {

			MemoryBlock* leadBlock = blocks[i];

			ul vacancySize = leadBlock->start - prevEnd - 1lu;

			// If a vacancy was found create a position struct
			if (vacancySize >= block->size) {
				block->start = prevEnd + 1lu;
				block->allocated = true;
				vacancyFound = true;
				freeSize -= block->size;
				blocks.insert(blocks.begin() + i, block);
				break;
			}
			prevEnd = leadBlock->start + leadBlock->size - 1lu;
		}

		// No vacancies were large enough to hold the new array
		if (!vacancyFound) {
			// TODO: implement cache formatting
			printf("Error: No vacancies found, cache needs formatting\n");
		}
	} else {
		if (block->allocated) {
			printf("Error: cannot free an currently allocated block\n");
		} else {
			printf("Error: not enough cache memory to allocate space for array\n");
		}
	}
}

void vat::DiskCache::freeBlock(MemoryBlock* block) {
	// TODO: implement binary search

	if (block->allocated && (DiskCache*)block->cache == this) {
		for (ul i = 0; i < blocks.size(); i++) {

			if (block == blocks[i]) {
				blocks.erase(blocks.begin() + i);
				block->allocated = false;
				break;
			}
		}
		if (block->allocated) {
			printf("Error: block not found when attempting to free\n");
		}
	} else {
		if (block->allocated && (DiskCache*)block->cache != this) {
			printf("Error: cannot free block from non-associated cache\n");
		}
	}
}

void vat::DiskCache::clearCache() {
	if (remove(path.c_str()) != 0) {
		printf("Error: unable to remove cache file: %s", strerror(errno));
	}
}

