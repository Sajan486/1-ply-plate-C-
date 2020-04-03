#ifndef UTIL_H_
#define UTIL_H_

#include <armadillo>
#include "Matrix.h"
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <sstream>


template<typename T>
arma::Mat<T>* rotateMat(arma::Mat<T>& mat);

template<typename T>
arma::Cube<T>* rotateCube(arma::Cube<T>& cube);

template<typename T>
arma::Mat<T> toArma(vat::Mat<T>* m);

template<typename T>
void toArma(vat::Mat<T>* m, arma::Mat<T>* newMat);

template<typename T>
void assertMatEquality(const arma::Mat<T>& a, const arma::Mat<T>& b, double tolerance = 1e-8);

void assertMatEquality(const arma::mat& a, const arma::mat& b, double tolerance = 1e-8);

void assertMatEquality(const arma::cx_mat& a, const arma::cx_mat& b, double tolerance = 1e-8);


template<typename T>
void assertMatEquality(arma::Mat<T>& a, vat::Mat<T>& b, double tolerance = 1e-8);

void assertMatEquality(arma::cx_mat& a, vat::CxdMat& b, double tolerance = 1e-8);

void assertFieldEquality(const arma::field<arma::cx_mat>& a, const arma::field<arma::cx_mat> b,
		double tolerance = 1e-8);

// Equate matrices
template<typename T>
bool approxEqual(vat::Mat<T>* m, arma::Mat<T>* a, double tolerance = 1e-8);

template<typename T>
bool approxEqual(vat::Mat<T>& m, vat::Mat<T>& a, double tolerance = 1e-8);

// Equate cubes
template<typename T>
bool approxEqual(vat::Mat<T>** m, arma::Cube<T>* a, int nSlices, double tolerance = 1e-8);

//Warning: This function is unstable and should only be used for debugging.
template<typename T>
void assertVatMatEquality(vat::Mat<T>& a, vat::Mat<T>& b, double tolerance = 1e-8);

void assertVatMatEquality(vat::CxdMat& a, vat::CxdMat& b, double tolerance = 1e-8);

template<typename T>
void assertCubeEquality(arma::Cube<T>& a, vat::Mat<T>** b, double tolerance = 1e-8);

void assertCubeEquality(arma::cx_cube& a, vat::CxdMat** b, double tolerance = 1e-8);

template<typename T>
cv::Mat matToCvMat(arma::Mat<T> armaMat) {

	unsigned long nRows = armaMat.n_rows;
	unsigned long nCols = armaMat.n_cols;

	cv::Mat cvMat(nRows, nCols, CV_64F);
	for (unsigned long row = 0lu; row < nRows; row++) {
		for (unsigned long col = 0lu; col < nCols; col++) {

			cvMat.at<double>(row, col) = armaMat(row, col);
		}
	}
	cv::Mat cvMatNorm;
	cv::normalize(cvMat, cvMatNorm, 0, 255, 32, CV_8UC1);

	return cvMatNorm;
}

template<typename T>
cv::Mat cx_matToCvMat(arma::Mat<std::complex<T>> armaMat) {

	arma::Mat<T> matMag(arma::size(armaMat));
	for (unsigned long row = 0lu; row < armaMat.n_rows; row++) {
		for (unsigned long col = 0lu; col < armaMat.n_cols; col++) {

			std::complex<T> val = armaMat(row, col);

			T real = val.real();
			T imag = val.imag();
			T mag  = sqrt(pow(real, 2) + pow(imag, 2));

			matMag(row, col) = mag;
		}
	}
	return matToCvMat(matMag);
}
template cv::Mat cx_matToCvMat(arma::Mat<std::complex<float>>);
template cv::Mat cx_matToCvMat(arma::Mat<std::complex<double>>);

template<typename T>
void savePlotData(int plotMode, std::vector<arma::Mat<std::complex<T>>> data,
		std::vector<std::string> dataTags,
		std::string directory,
		std::string validationFolder,
		bool validationMode,
		bool doublePrecision) {

	std::string plane;
	switch(plotMode) {
	case 1: plane = "XY"; break;
	case 2: plane = "YZ"; break;
	case 3: plane = "ZX"; break;
	}

	if (validationMode) {
		if (doublePrecision) {
			printf("\nvalidating (config::doublePrecision = true)...\n");
		} else {
			printf("\nvalidating (config::doublePrecision = false)...\n");
		}

	}
	for (int d = 0; d < data.size(); d++) {

		arma::Mat<std::complex<T>> cxMat = data[d];
		std::string  tag   = dataTags[d];

		cxMat.save(directory + "/" + tag + "_" + plane, arma::raw_ascii);
		arma::Mat<T> absMat = abs(cxMat);
		std::string matName = tag + "_MAG_" + plane;
		absMat.save(directory + "/" + matName, arma::raw_ascii);
		cv::Mat cvData = cx_matToCvMat(cxMat);
		cv::imwrite(directory + "/" + tag + "_" + plane + ".jpg", cvData);

		if (validationMode) {

			std::string validMatPath = validationFolder + "/" + matName;
			arma::Mat<T> validMat;
			validMat.load(validMatPath);
			arma::Mat<T> diff = validMat - absMat;
			arma::Mat<T> absDiff = abs(diff);
			arma::Mat<T> error = absDiff / validMat;

			T maxError = 100 * error.max();
			printf("%s differed from validation by %.2e%% (max)\n", matName.c_str(), maxError);
		}
	} printf("\n");
}
template void savePlotData(int, std::vector<arma::Mat<std::complex<float>>>,  std::vector<std::string>, std::string,
		std::string, bool, bool);
template void savePlotData(int, std::vector<arma::Mat<std::complex<double>>>, std::vector<std::string>, std::string,
		std::string, bool, bool);

template<typename T>
bool plotModeFound(std::vector<int> plotModes, int mode) {

	for (int i = 0; i < plotModes.size(); i++) {
		if (plotModes[i] == mode) return true;
	}
	return false;
}
bool plotModeFound(std::vector<int>, int);

std::string getTimePath(std::chrono::time_point<std::chrono::system_clock> time_point, std::string apriori,
		const std::string obj = "", const int iter = -1);

#endif
