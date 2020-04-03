#ifndef COMPPRESSUREFLUID_H_
#define COMPPRESSUREFLUID_H_

#include<stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include"armadillo"

using namespace std;
using namespace arma;

class CPFOutput {
public:
	cx_mat PR_FL;
};

CPFOutput CompPressureFluid(int plotMode, arma::cx_mat At, arma::cx_mat A1, arma::mat TransCoord_Btm,
		arma::mat IntrFcCoord_Top,
		arma::mat cell_coordt,
		double WaveNum_P,
		arma::vec,
		double NumSourcePt_Trans,
		double NumSourceTot);

#endif
