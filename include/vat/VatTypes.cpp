
#include "VatTypes.h"

vat::BlockDims::BlockDims() {
	x = 0lu;
	y = 0lu;
}

vat::BlockDims::BlockDims(ul _x, ul _y) {
	x = _x;
	y = _y;
}

vat::GridDims::GridDims() {
	x = 0lu;
	y = 0lu;
}

vat::GridDims::GridDims(ul _x, ul _y) {
	x = _x;
	y = _y;
}

vat::MemorySize::MemorySize() {
	size = 0;
	unit = B;
	byteSize = 0;
}

vat::MemorySize::MemorySize(double _size, vat::MemoryUnit _unit) {
	size = _size;
	unit = _unit;

	byteSize = (ul)(size * (double)unitSize(unit));
}

vat::ul vat::MemorySize::nBytes() {
	return byteSize;
}

double vat::MemorySize::sizeIn(MemoryUnit _unit) {
	return (double)byteSize / (double)unitSize(_unit);
}

vat::ul vat::MemorySize::unitSize(MemoryUnit _unit) {

	ul unitSz = 1lu;

	ul base;
	ul power;

	if (_unit < 5) {   // Decimal base
		base  = 1000;
		power = _unit;
	} else {           // Binary base
		base  = 1024;
		power = _unit - 4;
	}

	for (ul i = 0; i < power; i++) unitSz *= base;

	return unitSz;
}
