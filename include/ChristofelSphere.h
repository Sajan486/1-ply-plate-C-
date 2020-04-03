#ifndef _CHRISTOFELSPHERE_H
#define __CHRISTOFELSPHERE_H

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <armadillo>
#include "Config.h"
#include "Sphere.h"
#include <memory>

class ChristofelSphere {
public:
	ChristofelSphere(const double angTestPt);

	ChristofelSphere(ChristofelSphere const &) = delete;

	ChristofelSphere &operator= (ChristofelSphere const &) = delete;

	ChristofelSphere(ChristofelSphere &&obj) :
		CV(std::move(obj.CV)), FI(std::move(obj.FI)),
		sphere(std::move(obj.sphere)), nPts(obj.nPts)
	{}

	~ChristofelSphere();

	void solve(const arma::mat& C, double solidRho, double freq, double angTestPt);

	void save(const std::string& path = "");

	arma::field<arma::cx_mat>& getCV();
	arma::field<arma::cx_mat>& getFI();
	Sphere& getSphere();
	unsigned long getnPts();

private:
	std::unique_ptr<arma::field<arma::cx_mat>> CV;
	std::unique_ptr<arma::field<arma::cx_mat>> FI;
	std::unique_ptr<Sphere> sphere;
	unsigned long nPts;

	std::unique_ptr<arma::field<arma::cx_mat>>
		christofelSolution(const arma::mat& C, double nTrgX, double nTrgY, double nTrgZ, double solidRho, double freq);
};

#endif
