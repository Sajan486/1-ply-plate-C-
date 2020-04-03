#include "Util.h"

template<typename T>
arma::Mat<T>* rotateMat(arma::Mat<T>& mat) {
    const int rows = mat.n_rows,
              cols = mat.n_cols;
    arma::Mat<T>* rotMat = new arma::Mat<T>(cols, rows);
    for (int i = 0, z = cols-1; i < cols; i++, z--) {
        rotMat->row(i) = mat.col(z).t();
    }
    return rotMat;
}
template arma::Mat<float>*  rotateMat(arma::Mat<float>&);
template arma::Mat<double>* rotateMat(arma::Mat<double>&);
template arma::Mat<std::complex<float>>*  rotateMat(arma::Mat<std::complex<float>>&);
template arma::Mat<std::complex<double>>* rotateMat(arma::Mat<std::complex<double>>&);

template<typename T>
arma::Cube<T>* rotateCube(arma::Cube<T>& cube) {
    const int rows = cube.n_rows,
              cols = cube.n_cols;
    arma::Cube<T>* rotCube = new arma::Cube<T>(cols, rows, cube.n_slices);

    for (int i = 0; i < cube.n_slices; i++) {
        arma::Mat<T>* rotMat = rotateMat<T>(cube.slice(i));
        rotCube->slice(i) = *rotMat;
        delete rotMat;
    }
    return rotCube;
}
template arma::Cube<float>*  rotateCube(arma::Cube<float>&);
template arma::Cube<double>* rotateCube(arma::Cube<double>&);
template arma::Cube<std::complex<float>>*  rotateCube(arma::Cube<std::complex<float>>&);
template arma::Cube<std::complex<double>>* rotateCube(arma::Cube<std::complex<double>>&);

template<typename T>
arma::Mat<T> toArma(vat::Mat<T>* m) {
	unsigned long nRows = m->getN_Rows();
	unsigned long nCols = m->getN_Cols();

	T* rowMajorData = m->rows(0lu, nRows - 1lu);
	T* colMajorData = new T[nRows * nCols];

	for (vat::ul row = 0lu; row < nRows; row++) {
		for (vat::ul col = 0lu; col < nCols; col++) {
			colMajorData[row + nRows * col] = rowMajorData[col + nCols * row];
		}
	}

	arma::Mat<T>* armaMat = new arma::Mat<T>(colMajorData, nRows, nCols, false);

	delete[] rowMajorData;
	return *armaMat;
}
template arma::Mat<float>  toArma(vat::Mat<float>*  m);
template arma::Mat<double> toArma(vat::Mat<double>* m);
template arma::Mat<std::complex<float>>  toArma(vat::Mat<std::complex<float>>*  m);
template arma::Mat<std::complex<double>> toArma(vat::Mat<std::complex<double>>* m);


template<typename T>
void toArma(vat::Mat<T>* m, arma::Mat<T>* newMat) {
	unsigned long nRows = m->getN_Rows();
	unsigned long nCols = m->getN_Cols();
	if (nRows != newMat->n_rows || nCols != newMat->n_cols)
		throw std::runtime_error("Dimensions must match.");

	T* rowMajorData = m->rows(0lu, nRows - 1lu);

	for (vat::ul row = 0lu; row < nRows; row++) {
		for (vat::ul col = 0lu; col < nCols; col++) {
			newMat->at(row, col) = rowMajorData[col + nCols * row];
		}
	}
	delete[] rowMajorData;
}
template void toArma(vat::Mat<float>* m, arma::Mat<float>*);
template void toArma(vat::Mat<double>* m, arma::Mat<double>*);
template void toArma(vat::Mat<std::complex<float>>*  m, arma::Mat<std::complex<float>>*);
template void toArma(vat::Mat<std::complex<double>>* m, arma::Mat<std::complex<double>>*);


template<typename T>
void assertMatEquality(const arma::Mat<T>& a, const arma::Mat<T>& b, double tolerance) {
	assert(a.n_rows == b.n_rows && a.n_cols == b.n_cols);
	assert(approx_equal(a, b, "absdiff", tolerance));
}
template void assertMatEquality(const arma::Mat<float>&,  const arma::Mat<float>&,  double);
template void assertMatEquality(const arma::Mat<double>&, const arma::Mat<double>&, double);
template void assertMatEquality(const arma::Mat<std::complex<float>>&,  const arma::Mat<std::complex<float>>&,  double);
template void assertMatEquality(const arma::Mat<std::complex<double>>&, const arma::Mat<std::complex<double>>&, double);


void assertMatEquality(const arma::mat& a, const arma::mat& b, double tolerance) {
	assertMatEquality<double>(a, b, tolerance);
}

void assertMatEquality(const arma::cx_mat& a, const arma::cx_mat& b, double tolerance) {
	assertMatEquality<std::complex<double>>(a, b, tolerance);
}

template<typename T>
void assertMatEquality(arma::Mat<T>& a, vat::Mat<T>& b, double tolerance) {
	assert((unsigned long) a.n_rows == b.getN_Cols() && (unsigned long)a.n_cols == b.getN_Cols());
	arma::Mat<T> bToA = toArma(&b);
	assert(approx_equal(a, bToA, "absdiff", tolerance));
}
template void assertMatEquality(arma::Mat<float>&, vat::Mat<float>&,  double);
template void assertMatEquality(arma::Mat<double>&, vat::Mat<double>&, double);
template void assertMatEquality(arma::Mat<std::complex<float>>&, vat::Mat<std::complex<float>>&,  double);
template void assertMatEquality(arma::Mat<std::complex<double>>&, vat::Mat<std::complex<double>>&, double);


void assertMatEquality(arma::cx_mat& a, vat::CxdMat& b, double tolerance) {
	assertMatEquality<std::complex<double>>(a, b, tolerance);
}


void assertFieldEquality(const arma::field<arma::cx_mat>& a, const arma::field<arma::cx_mat> b,
		double tolerance) {
	assert (a.n_cols == b.n_cols && a.n_cols == b.n_rows);

	for (int i = 0; i < a.n_cols; i++) {
		for (int z = 0; z < a.n_rows; z++) {
			assert(approx_equal(a(z, i), b(z, i), "absdiff", tolerance));
		}
	}
}

template<typename T>
void assertCubeEquality(arma::Cube<T>& a, vat::Mat<T>** b, double tolerance) {
	assert((unsigned long) a.n_slices == sizeof(b) / sizeof(arma::Mat<T>*));
	for (int i = 0; i < (int)a.n_slices; i++) {
		arma::Mat<T> b_slice = toArma<T>(b[i]);
		assertMatEquality(a.slice(i), b_slice, tolerance);
	}
}
template void assertCubeEquality(arma::Cube<float>&, vat::Mat<float>**,  double);
template void assertCubeEquality(arma::Cube<double>&, vat::Mat<double>**, double);
template void assertCubeEquality(arma::Cube<std::complex<float>>&, vat::Mat<std::complex<float>>**,  double);
template void assertCubeEquality(arma::Cube<std::complex<double>>&, vat::Mat<std::complex<double>>**, double);

void assertCubeEquality(arma::cx_cube& a, vat::CxdMat** b, double tolerance) {
	assertCubeEquality<std::complex<double>>(a, b, tolerance);
}

template<typename T>
bool approxEqual(vat::Mat<T>* m, arma::Mat<T>* a, double tolerance) {

	arma::Mat<T> mToA = toArma(m);

	return approx_equal(mToA, *a, "absdiff", tolerance);
}
template bool approxEqual(vat::Mat<float>*,  arma::Mat<float>*,  double);
template bool approxEqual(vat::Mat<double>*, arma::Mat<double>*, double);
template bool approxEqual(vat::Mat<std::complex<float>>*,  arma::Mat<std::complex<float>>*,  double);
template bool approxEqual(vat::Mat<std::complex<double>>*, arma::Mat<std::complex<double>>*, double);


template<typename T>
bool approxEqual(vat::Mat<T>& m, vat::Mat<T>& a, double tolerance) {

	arma::Mat<T> mToA = toArma(&m);
	arma::Mat<T> aToA = toArma(&a);

	return approx_equal(mToA, aToA, "absdiff", tolerance);
}
template bool approxEqual(vat::Mat<float>&,  vat::Mat<float>&,  double);
template bool approxEqual(vat::Mat<double>&, vat::Mat<double>&, double);
template bool approxEqual(vat::Mat<std::complex<float>>&,  vat::Mat<std::complex<float>>&,  double);
template bool approxEqual(vat::Mat<std::complex<double>>&, vat::Mat<std::complex<double>>&, double);


template<typename T>
bool approxEqual(vat::Mat<T>** m, arma::Cube<T>* a, int nSlices, double tolerance) {

	bool equal = true;
	for (int slice = 0; slice < nSlices; slice++) {
		arma::Mat<T> mToA = toArma(m[slice]);

		if (!approx_equal(mToA, a->slice(slice), "absdiff", tolerance)) {
			equal = false; break;
		}
	}
	return equal;
}
template bool approxEqual(vat::Mat<float>**,  arma::Cube<float>*, int nSlices, double);
template bool approxEqual(vat::Mat<double>**, arma::Cube<double>*, int nSlices, double);
template bool approxEqual(vat::Mat<std::complex<float>>**,  arma::Cube<std::complex<float>>*, int nSlices,  double);
template bool approxEqual(vat::Mat<std::complex<double>>**, arma::Cube<std::complex<double>>*, int nSlices, double);


template<typename T>
void assertVatMatEquality(vat::Mat<T>& a, vat::Mat<T>& b, double tolerance) {
	assert(a.getN_Rows() == b.getN_Rows() && a.getN_Cols() == b.getN_Cols());
	assert(approxEqual<T>(a, b , tolerance));
}
template void assertVatMatEquality(vat::Mat<float>&, vat::Mat<float>&,  double);
template void assertVatMatEquality(vat::Mat<double>&, vat::Mat<double>&, double);
template void assertVatMatEquality(vat::Mat<std::complex<float>>&, vat::Mat<std::complex<float>>&,  double);
template void assertVatMatEquality(vat::Mat<std::complex<double>>&, vat::Mat<std::complex<double>>&, double);


std::string getTimePath(std::chrono::time_point<std::chrono::system_clock> time_point, std::string apriori,
		const std::string obj, const int iter) {
	std::stringstream ss;
	ss << std::chrono::system_clock::to_time_t(time_point);

	 return apriori + ss.str() +
			 (iter >= 0 ? std::to_string(iter) + "/" : "/") + obj;
}
