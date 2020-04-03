#ifndef SPHERE_H_
#define SPHERE_H_

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <armadillo>
#include <memory>

class Sphere {
public:
	Sphere(double angTestpt);

	Sphere(const Sphere &) = delete;
	Sphere &operator=(Sphere const &) = delete;

	~Sphere();

	void save(const std::string& path = "");

	arma::mat& getTheta();
	arma::mat& getPhi();
	arma::mat& getr();
	const unsigned long getnPts();

private:
	std::unique_ptr<arma::mat> theta;
	std::unique_ptr<arma::mat> phi;
	std::unique_ptr<arma::mat> r;

	unsigned long nPts;
};

#endif
