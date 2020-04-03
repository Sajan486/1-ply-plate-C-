#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "armadillo"
#include "CompPressureFluid.h"
#include "Config.h"


using namespace std;

CPFOutput CompPressureFluid(int plotMode, arma::cx_mat At, arma::cx_mat A1, arma::mat TransCoord_Btm,
		arma::mat IntrFcCoord_Top,
		arma::mat trgCoord,
		double WaveNum_P,
		arma::vec nTrgVec,
		double NumSourcePt_Trans,
		double NumSourceTot) {

	std::cout << "Begin CompPressureFluid\n";

	double NumTarget_x = nTrgVec(0);
	double NumTarget_y = nTrgVec(1);
	double NumTarget_z = nTrgVec(2);
	//double TotNumTarget = NumTarget_x * NumTarget_z;

	double TotNumTarget;
	if (plotMode == 2) TotNumTarget = NumTarget_y * NumTarget_z;
	if (plotMode == 3) TotNumTarget = NumTarget_x * NumTarget_z;

	complex<double> img(0, 1);
	double CoordCentSourcePt_x = 0, CoordCentSourcePt_y = 0, CoordCentSourcePt_z = 0;
	arma::mat R(NumSourcePt_Trans, TotNumTarget);
	arma::cx_mat P(TotNumTarget, NumSourcePt_Trans);
	arma::cx_mat P1(1, TotNumTarget);

	for (int is = 0; is< NumSourcePt_Trans;is++) {
		
		CoordCentSourcePt_x = TransCoord_Btm(0, is);
		CoordCentSourcePt_y = TransCoord_Btm(1, is);
		CoordCentSourcePt_z = TransCoord_Btm(2, is);
		for (int jt = 0; jt < TotNumTarget; jt++) {
			R(is, jt) = sqrt(pow((trgCoord(jt, 0) - CoordCentSourcePt_x), 2) + pow((trgCoord(jt, 1) - CoordCentSourcePt_y), 2) + pow((trgCoord(jt, 2) - CoordCentSourcePt_z), 2));
		}
	}

	for (int jt = 0; jt <TotNumTarget; jt++) {
		P1(0, jt) = 0.0;
		for (int is = 0; is< NumSourcePt_Trans; is++) {
			P(jt, is) = At(0, is)*(exp(img*WaveNum_P*R(is, jt))) / (R(is, jt));
			P1(0, jt) = P1(0, jt) + P(jt, is);
		}
	}

	CoordCentSourcePt_x = 0, CoordCentSourcePt_y = 0, CoordCentSourcePt_z = 0;
	arma::mat R_1(NumSourceTot, TotNumTarget);
	arma::cx_mat P_1(TotNumTarget, NumSourceTot);
	arma::cx_mat P2(1, TotNumTarget);
	for (int is = 0; is< NumSourceTot; is++) {
		CoordCentSourcePt_x = IntrFcCoord_Top(0, is);
		CoordCentSourcePt_y = IntrFcCoord_Top(1, is);
		CoordCentSourcePt_z = IntrFcCoord_Top(2, is);

		for (int jt = 0; jt< TotNumTarget; jt++) {
			R_1(is, jt) = sqrt(pow((trgCoord(jt, 0) - CoordCentSourcePt_x), 2) + pow((trgCoord(jt, 1) - CoordCentSourcePt_y), 2) + pow((trgCoord(jt, 2) - CoordCentSourcePt_z), 2));
		}
	}

	for (int jt = 0; jt< TotNumTarget; jt++) {
		P2(0, jt) = 0.0;
		for (int is = 0; is< NumSourceTot; is++) {

			P_1(jt, is) = A1(0, is)*(exp(img*WaveNum_P*R_1(is, jt))) / (R_1(is, jt));
			P2(0, jt) = P2(0, jt) + P_1(jt, is);
		}
	}
	
	arma::cx_mat PR_FL(1, TotNumTarget);
	for (int jt = 0; jt< TotNumTarget; jt++) {
		PR_FL(0, jt) = P1(0, jt) + P2(0, jt);
	}

	CPFOutput value;
	value.PR_FL = PR_FL;

	std::cout << "End CompPressureFluid\n";
	return value;
}

