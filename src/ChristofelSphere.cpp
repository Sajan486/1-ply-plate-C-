#include "ChristofelSphere.h"

ChristofelSphere::ChristofelSphere(const double angTestPt) {
	nPts = (unsigned long) round(360 / angTestPt);
	sphere = std::unique_ptr<Sphere> (new Sphere(angTestPt));

	const unsigned long npts = sphere->getnPts();
	CV = std::unique_ptr<arma::field<arma::cx_mat>> (new arma::field<arma::cx_mat>(npts, npts));
	FI = std::unique_ptr<arma::field<arma::cx_mat>> (new arma::field<arma::cx_mat>(npts, npts));
}

ChristofelSphere::~ChristofelSphere() {}

void ChristofelSphere::solve(const arma::mat& C, double solidRho, double freq, double angTestPt) {

	for (int i = 0; i < sphere->getnPts(); i++) {
		for (int j = 0; j < sphere->getnPts(); j++) {

			double x = sphere->getr().at(i, j) * cos(sphere->getTheta().at(i, j)) * cos(sphere->getPhi().at(i, j));
			double y = sphere->getr().at(i, j) * sin(sphere->getTheta().at(i, j)) * cos(sphere->getPhi().at(i, j));
			double z = sphere->getr().at(i, j) * sin(sphere->getPhi().at(i, j));

			double nPtsX = x / (sqrt(x * x + y * y + z * z));
			double nPtsY = y / (sqrt(x * x + y * y + z * z));
			double nPtsZ = z / (sqrt(x * x + y * y + z * z));

			std::unique_ptr<arma::field<arma::cx_mat>> res = christofelSolution(C, nPtsX, nPtsY, nPtsZ, solidRho, freq);
			FI->at(i, j) = res->at(0, 0);
			CV->at(i, j) = res->at(0, 1);
		}
	}
}

std::unique_ptr<arma::field<arma::cx_mat>>
ChristofelSphere::christofelSolution(const arma::mat&  C,
		                             double     nTrgX,
		                             double     nTrgY,
		                             double     nTrgZ,
		                             double     solidRho,
		                             double     freq) {

	// Christoffel Acoustic Tensor Components
	double L11 = (C(0, 0) * pow(nTrgX, 2)) + (C(5, 5) * pow(nTrgY, 2)) + (C(4, 4) * pow(nTrgZ, 2)) +       (2 * C(4, 5) * (nTrgY*nTrgZ)) + (2 * C(0, 4) *       (nTrgX*nTrgZ)) + (2 * C(0, 5) *       (nTrgX*nTrgY));
	double L12 = (C(0, 5) * pow(nTrgX, 2)) + (C(1, 5) * pow(nTrgY, 2)) + (C(3, 4) * pow(nTrgZ, 2)) + ((C(0, 1) + C(5, 5))*(nTrgX*nTrgY)) + ((C(0, 3) + C(4, 5))*(nTrgX*nTrgZ)) + ((C(3, 5) + C(1, 4))*(nTrgY*nTrgZ));
	double L13 = (C(0, 4) * pow(nTrgX, 2)) + (C(3, 5) * pow(nTrgY, 2)) + (C(2, 4) * pow(nTrgZ, 2)) +  ((C(0, 3)+ C(4, 5))*(nTrgX*nTrgY)) + ((C(0, 2) + C(4, 4))*(nTrgX*nTrgZ)) + ((C(2, 5) + C(3, 4))*(nTrgY*nTrgZ));
	double L22 = (C(5, 5) * pow(nTrgX, 2)) + (C(1, 1) * pow(nTrgY, 2)) + (C(3, 3) * pow(nTrgZ, 2)) +       (2 * C(1, 5) * (nTrgX*nTrgY)) + (2 * C(3, 5) *       (nTrgX*nTrgZ)) + (2 * C(1, 3) *       (nTrgY*nTrgZ));
	double L23 = (C(4, 5) * pow(nTrgX, 2)) + (C(1, 3) * pow(nTrgY, 2)) + (C(2, 3) * pow(nTrgZ, 2)) + ((C(3, 5) + C(1, 4))*(nTrgX*nTrgY)) + ((C(2, 5) + C(3, 4))*(nTrgX*nTrgZ)) + ((C(1, 2) + C(3, 3))*(nTrgY*nTrgZ));
	double L33 = (C(4, 4) * pow(nTrgX, 2)) + (C(3, 3) * pow(nTrgY, 2)) + (C(2, 2) * pow(nTrgZ, 2)) +       (2 * C(3, 4) * (nTrgX*nTrgY)) + (2 * C(2, 4) *       (nTrgX*nTrgZ)) + (2 * C(2, 3) *       (nTrgY*nTrgZ));
	double L21 = L12;
	double L31 = L13;
	double L32 = L23;

	// Put all the L values into the matrix LIJ
	double n = solidRho * pow(freq, 2);
	arma::cx_mat LIJ(3, 3);						         // Christoffel Acoustic Tensor

	LIJ(0, 0) = L11 / n;
	LIJ(0, 1) = L12 / n;
	LIJ(0, 2) = L13 / n;
	LIJ(1, 0) = L21 / n;
	LIJ(1, 1) = L22 / n;
	LIJ(1, 2) = L23 / n;
	LIJ(2, 0) = L31 / n;
	LIJ(2, 1) = L32 / n;
	LIJ(2, 2) = L33 / n;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			double val = abs(LIJ(i, j));
			if (val < 1e-8) LIJ(i, j) = 0.0;
		}
	}

	arma::vec kinv(3, 1), kinv2;
	arma::cx_mat Fi(3, 3), Fi2, cv(1, 3);
	arma::uvec valindx(3, 1);

	arma::eig_sym (kinv2, Fi2, LIJ, "std");  //fff eig_gen
	valindx = arma::stable_sort_index(kinv2);

	for (int j = 0; j < 3; j++) {
		kinv(j, 0) = kinv2(valindx(j, 0), 0);
		Fi.col(j) = Fi2.col(valindx(j, 0));
	}

	for (int i = 0; i < 3; i++) {
		cv(0, i) = sqrt((freq * freq) * kinv(i));
	}

	std::unique_ptr<arma::field<arma::cx_mat>> output (new arma::field<arma::cx_mat>(1, 2));

	output->at(0, 0) = Fi;
	output->at(0, 1) = cv;
	return output;
}

void ChristofelSphere::save(const std::string& path) {
	CV->save(path + "CV", arma::csv_ascii);
	FI->save(path + "FI", arma::csv_ascii);
	sphere->save(path);
}

arma::field<arma::cx_mat>& ChristofelSphere::getCV() {
	return *CV;
}

arma::field<arma::cx_mat>& ChristofelSphere::getFI() {
	return *FI;
}

Sphere& ChristofelSphere::getSphere() {
	return *sphere;
}
