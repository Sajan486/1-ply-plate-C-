#include "Sphere.h"

Sphere::Sphere (double angTestPt) {
	nPts = (unsigned long) round(360 / angTestPt) + 1;

	theta = std::unique_ptr<arma::mat> (new arma::mat(nPts, nPts));
	phi   = std::unique_ptr<arma::mat> (new arma::mat(nPts, nPts));
	r     = std::unique_ptr<arma::mat> (new arma::mat(nPts, nPts));

	const double dT = 2 * M_PI / (double) (nPts - 1lu);  // Angle step of Theta
	const double dP = M_PI / (double) (nPts - 1lu);	  // Angle step of Phi

	double t, p; // local values of theta and phi

	for (int i = 0; i < nPts; i++) {
		for (int j = 0; j < nPts; j++) {

			if (j == 0 || j == nPts - 1lu) t = 0;
			else {
				t = i * dT + M_PI;
				if (t > M_PI) t = t - 2 * M_PI;
			}
			p = -0.5 * M_PI + j * dP;

			r->at    (j, i) = 1;
			theta->at(j, i) = t; // Populate Theta
			phi->at  (j, i) = p; // Populate Phi

		}
	}
}

Sphere::~Sphere() {}

void Sphere::save(const std::string& path) {
	theta->save(path + "sphere/theta", arma::csv_ascii);
	phi->save(path + "sphere/phi", arma::csv_ascii);
	r->save(path + "sphere/r", arma::csv_ascii);
}

arma::mat& Sphere::getTheta() {
	return *theta;
}

arma::mat& Sphere::getPhi() {
	return *phi;
}

arma::mat& Sphere::getr() {
	return *r;
}

const unsigned long Sphere::getnPts() {
	return nPts;
}
