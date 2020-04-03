#ifndef COMPWAVEFIELDSOLID_H_
#define COMPWAVEFIELDSOLID_H_

#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <armadillo>
#include <memory>
#include "CompPressureFluid.h"
#include "SolidGreen.h"

using namespace std;
using namespace arma;

class CWFSOutput {
public:
	cx_mat S331, S311, S321, S111, u1, u2, u3;
};

CWFSOutput CompWaveFieldSolid(int, arma::cx_mat&, arma::cx_mat&, arma::field<arma::mat>*, arma::mat&, arma::mat&,
		ChristofelSphere&, arma::vec, arma::vec, double, double, double, double, vat::DiskCache*, Config&);

#endif
