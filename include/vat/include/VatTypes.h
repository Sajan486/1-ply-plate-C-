#include <stdio.h>
#include <string>
#include <complex>

namespace vat {

typedef unsigned long ul;

enum RotationType {
    CCW = 0,
    CW = 1,
    Flip = 2
};

enum Order {
	RowMajor = 0,
	ColumnMajor = 1
};

struct BlockDims {

	BlockDims();
	BlockDims(ul _x, ul _y);
	ul x;
	ul y;
};

struct GridDims {

	GridDims();
	GridDims(ul _x, ul _y);
	ul x;
	ul y;
};

enum MemoryType {
	Static = 0,
	Dynamic = 1
};

enum MemoryUnit {
	B   = 0,
	KB  = 1,
	MB  = 2,
	GB  = 3,
	TB  = 4,
	KiB = 5,
	MiB = 6,
	GiB = 7,
	TiB = 8
};

struct MemoryBlock {
	ul    size;
	ul    start;
	bool  allocated;
	void* cache;
};

class MemorySize {
public:

	MemorySize();
	MemorySize(double _size, MemoryUnit _unit);

	ul nBytes();
	double sizeIn(MemoryUnit unit);

private:

	double     size;
	MemoryUnit unit;
	ul         byteSize;

	ul unitSize(MemoryUnit _unit);
};

}
