#include "CompWaveFieldSolid.h"


using namespace std;
using namespace arma;


CWFSOutput CompWaveFieldSolid(int                       plotMode,
							  cx_mat&                   A_1s,
							  cx_mat&                   A2s,
							  field<mat>*               IntrFcCoord_Btm,
							  mat&                      sweepTrgCoord,
							  mat&                      C,
							  ChristofelSphere&   cSphere,
							  arma::vec                 nTrgVec,
							  arma::vec                 nSweepTrgVec,
							  double                    Solid_rho,
							  double                    freq,
							  double                    dTheta,
							  double                    dPhi,
							  vat::DiskCache*			cache,
							  Config&					config) {

	double tolerance = 1e-8;

	/*
	 * Note that though SolidGreen is templated, the precision is forced by c
	 * configuration settings, until a later version
	 */

	printf("Begin CompWaveFieldSolid\n");

	double nTrgX = nTrgVec(0);
	double nTrgY = nTrgVec(1);
	double nTrgZ = nTrgVec(2);
	double nSweepTrgX = nSweepTrgVec(0);
	double nSweepTrgY = nSweepTrgVec(1);
	CWFSOutput value;

	if (plotMode == 1) {

		double nTrg = nTrgX * nTrgY;

		// Calculation Starts for t1 (Point Sources at Interface 1 - Bottom)
		cx_mat TR(1, 3 * nSweepTrgX * nSweepTrgY);

		double srcIdx = (nTrgX * nTrgY + 1) / 2;
		double xSourceCent = IntrFcCoord_Btm->at(0, 0)(0, srcIdx - 1);
		double ySourceCent = IntrFcCoord_Btm->at(0, 0)(1, srcIdx - 1);
		double zSourceCent = IntrFcCoord_Btm->at(0, 0)(2, srcIdx - 1);

		for (unsigned long h = 0; h < nSweepTrgY; h++) {
			for (unsigned long i = 0; i < nSweepTrgX; i++) {  // total no. of target points

				unsigned long Sw_TrgLoc = i + h * nSweepTrgX;
				unsigned long k         = Sw_TrgLoc;

				TR.col(3 * k + 0) = sweepTrgCoord(Sw_TrgLoc, 0) - xSourceCent;
				TR.col(3 * k + 1) = sweepTrgCoord(Sw_TrgLoc, 1) - ySourceCent;
				TR.col(3 * k + 2) = sweepTrgCoord(Sw_TrgLoc, 2) - zSourceCent;

				for (int i = 0; i < 3; i++) {
					std::complex<double> cmplx = TR(0, 3 * k + i);
					double real = abs(cmplx.real());
					if (real < tolerance) {
						TR(0, 3 * k + i) = std::complex<double>(0, 0);
					}
				}
			}
		}
		std::unique_ptr<SolidGreen<double>> sg (new SolidGreen<double>(nSweepTrgX, nSweepTrgY, cache));
		sg->solve(TR, C, cSphere, Solid_rho, freq, dTheta, dPhi, config);

		arma::cx_mat S33_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t1   (1, nTrg, arma::fill::zeros);

		complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgY; h++) {
			for (int i = 0; i < nTrgX; i++) {
				for (int j = 0; j < nTrgY; j++) {
					for (int k = 0; k < nTrgX; k++) {

						unsigned long index = k + j * nTrgX;
						unsigned long index1 = nTrgX + i - k - 1;
						unsigned long index2 = nTrgY - h + j - 1;
						unsigned long index3 = i + h * nTrgX;

						S33t=(A_1s(0,index)*sg->s33->at(index1,index2,0))+(A_1s(1,index)*sg->s33->at(index1,index2,1))+(A_1s(2,index)*sg->s33->at(index1,index2,2));
						S31t=(A_1s(0,index)*sg->s13->at(index1,index2,0))+(A_1s(1,index)*sg->s13->at(index1,index2,1))+(A_1s(2,index)*sg->s13->at(index1,index2,2));
						S32t=(A_1s(0,index)*sg->s23->at(index1,index2,0))+(A_1s(1,index)*sg->s23->at(index1,index2,1))+(A_1s(2,index)*sg->s23->at(index1,index2,2));
						S11t=(A_1s(0,index)*sg->s11->at(index1,index2,0))+(A_1s(1,index)*sg->s11->at(index1,index2,1))+(A_1s(2,index)*sg->s11->at(index1,index2,2));

						S33_t1(0lu, index3) += S33t;
						S31_t1(0lu, index3) += S31t;
						S32_t1(0lu, index3) += S32t;
						S11_t1(0lu, index3) += S11t;

						u1t=(A_1s(0,index)*sg->u1->at(index1,index2,0))+(A_1s(1,index)*sg->u1->at(index1,index2,1))+(A_1s(2,index)*sg->u1->at(index1,index2,2));
						u2t=(A_1s(0,index)*sg->u2->at(index1,index2,0))+(A_1s(1,index)*sg->u2->at(index1,index2,1))+(A_1s(2,index)*sg->u2->at(index1,index2,2));
						u3t=(A_1s(0,index)*sg->u3->at(index1,index2,0))+(A_1s(1,index)*sg->u3->at(index1,index2,1))+(A_1s(2,index)*sg->u3->at(index1,index2,2));

						u1_t1(0lu, index3) += u1t;
						u2_t1(0lu, index3) += u2t;
						u3_t1(0lu, index3) += u3t;
					}
				}
			}
		}

		// Calculation Starts for t2(Point Sources at Interface 2 - Top)
		// Implementing SISMAG to obtain Green's function for t2

		arma::cx_cube u1t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube u2t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube u3t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube S11t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube S33t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube S31t2(nSweepTrgX, nSweepTrgY, 3);
		arma::cx_cube S32t2(nSweepTrgX, nSweepTrgY, 3);


		for (unsigned long i = 0; i < nSweepTrgX; i++) {
			for (unsigned long j = 0; j < nSweepTrgY; j++) {

				// For Force along 1 direction
				u1t2.at(i, j , 0) = (sg->u1->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				u2t2.at(i, j , 0) = (sg->u2->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				u3t2.at(i, j , 0) = -(sg->u3->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				S11t2.at(i, j , 0) = (sg->s11->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				S33t2.at(i, j , 0) = (sg->s33->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				S31t2.at(i, j , 0) = -(sg->s13->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));
				S32t2.at(i, j , 0) = -(sg->s23->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 0));

				// For Force along 2 direction
				u1t2.at(i, j , 1) = (sg->u1->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				u2t2.at(i, j , 1) = (sg->u2->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				u3t2.at(i, j , 1) = -(sg->u3->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				S11t2.at(i, j , 1) = (sg->s11->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				S33t2.at(i, j , 1) = (sg->s33->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				S31t2.at(i, j , 1) = -(sg->s13->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));
				S32t2.at(i, j , 1) = -(sg->s23->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 1));

				// For Force along 3 direction
				u1t2.at(i, j , 2) = -(sg->u1->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				u2t2.at(i, j , 2) = -(sg->u2->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				u3t2.at(i, j , 2) = (sg->u3->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				S11t2.at(i, j , 2) = -(sg->s11->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				S33t2.at(i, j , 2) = -(sg->s33->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				S31t2.at(i, j , 2) = (sg->s13->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));
				S32t2.at(i, j , 2) = (sg->s23->at(nSweepTrgX -1- i, nSweepTrgY -1- j, 2));

			}
		}

		arma::cx_mat S33_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t2(1, nTrg, arma::fill::zeros);

		//complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgY; h++) {
			for (int i = 0; i < nTrgX; i++) {
				for (int j = 0; j < nTrgY; j++) {
					for (int k = 0; k < nTrgX; k++) {

						unsigned long index = k + j * nTrgX;
						unsigned long index1 = nTrgX + i - k - 1;
						unsigned long index2 = nTrgY - h + j - 1;
						unsigned long index3 = i + h * nTrgX;

						S33t = (A2s(0, index)*S33t2.at(index1, index2, 0)) + (A2s(1, index)*S33t2.at(index1, index2, 1)) + (A2s(2, index)*S33t2.at(index1, index2, 2));
						S31t = (A2s(0, index)*S31t2.at(index1, index2, 0)) + (A2s(1, index)*S31t2.at(index1, index2, 1)) + (A2s(2, index)*S31t2.at(index1, index2, 2));
						S32t = (A2s(0, index)*S32t2.at(index1, index2, 0)) + (A2s(1, index)*S32t2.at(index1, index2, 1)) + (A2s(2, index)*S32t2.at(index1, index2, 2));
						S11t = (A2s(0, index)*S11t2.at(index1, index2, 0)) + (A2s(1, index)*S11t2.at(index1, index2, 1)) + (A2s(2, index)*S11t2.at(index1, index2, 2));

						S33_t2(0lu, index3) += S33t;
						S31_t2(0lu, index3) += S31t;
						S32_t2(0lu, index3) += S32t;
						S11_t2(0lu, index3) += S11t;

						u1t = (A2s(0, index)*u1t2.at(index1, index2, 0)) + (A2s(1, index)*u1t2.at(index1, index2, 1)) + (A2s(2, index)*u1t2.at(index1, index2, 2));
						u2t = (A2s(0, index)*u2t2.at(index1, index2, 0)) + (A2s(1, index)*u2t2.at(index1, index2, 1)) + (A2s(2, index)*u2t2.at(index1, index2, 2));
						u3t = (A2s(0, index)*u3t2.at(index1, index2, 0)) + (A2s(1, index)*u3t2.at(index1, index2, 1)) + (A2s(2, index)*u3t2.at(index1, index2, 2));

						u1_t2(0lu, index3) += u1t;
						u2_t2(0lu, index3) += u2t;
						u3_t2(0lu, index3) += u3t;
					}
				}
			}
		}

		arma::cx_mat S331(1, nTrg, arma::fill::zeros);
		arma::cx_mat S311(1, nTrg, arma::fill::zeros);
		arma::cx_mat S321(1, nTrg, arma::fill::zeros);
		arma::cx_mat S111(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3(1, nTrg, arma::fill::zeros);

		for (unsigned long i = 0; i < nTrg; i++) {

			S331(0lu, i) = S33_t1(0lu, i) + S33_t2(0lu, i);
			S311(0lu, i) = S31_t1(0lu, i) + S31_t2(0lu, i);
			S321(0lu, i) = S32_t1(0lu, i) + S32_t2(0lu, i);
			S111(0lu, i) = S11_t1(0lu, i) + S11_t2(0lu, i);
			u1(0lu, i) = u1_t1(0lu, i) + u1_t2(0lu, i);
			u2(0lu, i) = u2_t1(0lu, i) + u2_t2(0lu, i);
			u3(0lu, i) = u3_t1(0lu, i) + u3_t2(0lu, i);

		}

		value.S331 = S331;
		value.S311 = S311;
		value.S321 = S321;
		value.S111 = S111;
		value.u1 = u1;
		value.u2 = u2;
		value.u3 = u3;

	}
	if (plotMode == 2) {

		double nTrg = nTrgY * nTrgZ;
		double nSweepTrg = nSweepTrgY * nTrgZ;

		arma::cx_mat TR(1, 3 * nTrgX * nSweepTrgY * nTrgZ);
		for (int g = 0; g < nTrgX; g++) {

			double SrcIndex = g * nTrgY + (nTrgY + 1) / 2;

			// Coordinate of the central point source just 'Source_EqivR' distance below the interface
			double xSourceCent = IntrFcCoord_Btm->at(0, 0)(0, SrcIndex - 1);
			double ySourceCent = IntrFcCoord_Btm->at(0, 0)(1, SrcIndex - 1);
			double zSourceCent = IntrFcCoord_Btm->at(0, 0)(2, SrcIndex - 1);

			for (unsigned long h = 0; h < nTrgZ; h++) {
				for (unsigned long i = 0; i < nSweepTrgY; i++) {  // total no. of target points

					unsigned long Sw_TrgLoc = i + h * nSweepTrgY;
					unsigned long k         = Sw_TrgLoc + g * nSweepTrgY * nTrgZ;

					TR.col(3 * k + 0) = sweepTrgCoord(Sw_TrgLoc, 0) - xSourceCent;
					TR.col(3 * k + 1) = sweepTrgCoord(Sw_TrgLoc, 1) - ySourceCent;
					TR.col(3 * k + 2) = sweepTrgCoord(Sw_TrgLoc, 2) - zSourceCent;

					for (int i = 0; i < 3; i++) {
						std::complex<double> cmplx = TR(0, 3 * k + i);
						double real = abs(cmplx.real());
						if (real < tolerance) {
							TR(0, 3 * k + i) = std::complex<double>(0, 0);
						}
					}
				}
			}
		}

		std::unique_ptr<SolidGreen<double>> sg (new SolidGreen<double>(nSweepTrgY * nTrgZ, nTrgX, cache));
		sg->solve(TR, C, cSphere, Solid_rho, freq, dTheta, dPhi, config);

		arma::cx_mat S33_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t1   (1, nTrg, arma::fill::zeros);

		complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgZ; h++) {
			for (int i = 0; i < nTrgY; i++) {				// for target points in Y - Z plane
				for (int j = 0; j < nTrgX; j++) {
					for (int k = 0; k < nTrgY; k++) {		// for source points in X - Y plane

						double index  = k + j * nTrgY;
						double index1 = i + h * nSweepTrgY + nTrgY - k - 1;
						double index2 = i + h * nTrgY;

						//For Stress
						S33t = (A_1s(0, index)*sg->s33->at(index1, j, 0)) + (A_1s(1, index)*sg->s33->at(index1, j, 1)) + (A_1s(2, index)*sg->s33->at(index1, j, 2));
						S31t = (A_1s(0, index)*sg->s13->at(index1, j, 0)) + (A_1s(1, index)*sg->s13->at(index1, j, 1)) + (A_1s(2, index)*sg->s13->at(index1, j, 2));
						S32t = (A_1s(0, index)*sg->s23->at(index1, j, 0)) + (A_1s(1, index)*sg->s23->at(index1, j, 1)) + (A_1s(2, index)*sg->s23->at(index1, j, 2));
						S11t = (A_1s(0, index)*sg->s11->at(index1, j, 0)) + (A_1s(1, index)*sg->s11->at(index1, j, 1)) + (A_1s(2, index)*sg->s11->at(index1, j, 2));

						S33_t1(0, index2) = S33_t1(0, index2) + S33t;
						S31_t1(0, index2) = S31_t1(0, index2) + S31t;
						S32_t1(0, index2) = S32_t1(0, index2) + S32t;
						S11_t1(0, index2) = S11_t1(0, index2) + S11t;

						//For Displacement
						u1t = (A_1s(0, index)*sg->u1->at(index1, j, 0)) + (A_1s(1, index)*sg->u1->at(index1, j, 1)) + (A_1s(2, index)*sg->u1->at(index1, j, 2));
						u2t = (A_1s(0, index)*sg->u2->at(index1, j, 0)) + (A_1s(1, index)*sg->u2->at(index1, j, 1)) + (A_1s(2, index)*sg->u2->at(index1, j, 2));
						u3t = (A_1s(0, index)*sg->u3->at(index1, j, 0)) + (A_1s(1, index)*sg->u3->at(index1, j, 1)) + (A_1s(2, index)*sg->u3->at(index1, j, 2));

						u1_t1(0, index2) = u1_t1(0, index2) + u1t;
						u2_t1(0, index2) = u2_t1(0, index2) + u2t;
						u3_t1(0, index2) = u3_t1(0, index2) + u3t;
					}
				}
			}
		}

		// Calculation Starts for t2(Point Sources at Interface 2 - Top)
		// Implementing SISMAG to obtain Green's function for t2

		cx_cube u1t2  (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube u2t2  (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube u3t2  (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube S11t2 (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube S33t2 (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube S31t2 (nSweepTrgY * nTrgZ, nTrgX, 3);
		cx_cube S32t2 (nSweepTrgY * nTrgZ, nTrgX, 3);

		for (unsigned long i = 0; i < nSweepTrgY * nTrgZ; i++) {
			for (unsigned long j = 0; j < nTrgX; j++) {

				// For Force along 1 direction
				u1t2.at(i, j, 0) = (sg->u1->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				u2t2.at(i, j, 0) = (sg->u2->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				u3t2.at(i, j, 0) = -(sg->u3->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				S11t2.at(i, j, 0) = (sg->s11->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				S33t2.at(i, j, 0) = (sg->s33->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				S31t2.at(i, j, 0) = -(sg->s13->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));
				S32t2.at(i, j, 0) = -(sg->s23->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 0));

				// For Force along 2 direction
				u1t2.at(i, j, 1) = (sg->u1->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				u2t2.at(i, j, 1) = (sg->u2->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				u3t2.at(i, j, 1) = -(sg->u3->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				S11t2.at(i, j, 1) = (sg->s11->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				S33t2.at(i, j, 1) = (sg->s33->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				S31t2.at(i, j, 1) = -(sg->s13->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));
				S32t2.at(i, j, 1) = -(sg->s23->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 1));

				// For Force along 3 direction
				u1t2.at(i, j, 2) = -(sg->u1->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				u2t2.at(i, j, 2) = -(sg->u2->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				u3t2.at(i, j, 2) = (sg->u3->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				S11t2.at(i, j, 2) = -(sg->s11->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				S33t2.at(i, j, 2) = -(sg->s33->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				S31t2.at(i, j, 2) = (sg->s13->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));
				S32t2.at(i, j, 2) = (sg->s23->at(nSweepTrgY * nTrgZ -1- i, nTrgX -1- j, 2));

			}
		}
		arma::cx_mat S33_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t2(1, nTrg, arma::fill::zeros);

		//complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgZ; h++) {
			for (int i = 0; i < nTrgY; i++) {				// for target points in Y - Z plane
				for (int j = 0; j < nTrgX; j++) {
					for (int k = 0; k < nTrgY; k++) {		// for source points in X - Y plane

						double index = k + j * nTrgY;
						double index1 = i + h * nSweepTrgY + nTrgY - k - 1;
						double index2 = i + h * nTrgY;

						//For Stress
						S33t = (A2s(0, index)*sg->s33->at(index1, j, 0)) + (A2s(1, index)*sg->s33->at(index1, j, 1)) + (A2s(2, index)*sg->s33->at(index1, j, 2));
						S31t = (A2s(0, index)*sg->s13->at(index1, j, 0)) + (A2s(1, index)*sg->s13->at(index1, j, 1)) + (A2s(2, index)*sg->s13->at(index1, j, 2));
						S32t = (A2s(0, index)*sg->s23->at(index1, j, 0)) + (A2s(1, index)*sg->s23->at(index1, j, 1)) + (A2s(2, index)*sg->s23->at(index1, j, 2));
						S11t = (A2s(0, index)*sg->s11->at(index1, j, 0)) + (A2s(1, index)*sg->s11->at(index1, j, 1)) + (A2s(2, index)*sg->s11->at(index1, j, 2));

						S33_t2(0, index2) = S33_t2(0, index2) + S33t;
						S31_t2(0, index2) = S31_t2(0, index2) + S31t;
						S32_t2(0, index2) = S32_t2(0, index2) + S32t;
						S11_t2(0, index2) = S11_t2(0, index2) + S11t;

						//For Displacement
						u1t = (A2s(0, index)*sg->u1->at(index1, j, 0)) + (A2s(1, index)*sg->u1->at(index1, j, 1)) + (A2s(2, index)*sg->u1->at(index1, j, 2));
						u2t = (A2s(0, index)*sg->u2->at(index1, j, 0)) + (A2s(1, index)*sg->u2->at(index1, j, 1)) + (A2s(2, index)*sg->u2->at(index1, j, 2));
						u3t = (A2s(0, index)*sg->u3->at(index1, j, 0)) + (A2s(1, index)*sg->u3->at(index1, j, 1)) + (A2s(2, index)*sg->u3->at(index1, j, 2));

						u1_t2(0, index2) = u1_t2(0, index2) + u1t;
						u2_t2(0, index2) = u2_t2(0, index2) + u2t;
						u3_t2(0, index2) = u3_t2(0, index2) + u3t;
					}
				}
			}
		}



		arma::cx_mat S331(1, nTrg, arma::fill::zeros);
		arma::cx_mat S311(1, nTrg, arma::fill::zeros);
		arma::cx_mat S321(1, nTrg, arma::fill::zeros);
		arma::cx_mat S111(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3(1, nTrg, arma::fill::zeros);

		for (unsigned long i = 0; i < nTrg; i++) {

			S331(0lu, i) = S33_t1(0lu, i) + S33_t2(0lu, i);
			S311(0lu, i) = S31_t1(0lu, i) + S31_t2(0lu, i);
			S321(0lu, i) = S32_t1(0lu, i) + S32_t2(0lu, i);
			S111(0lu, i) = S11_t1(0lu, i) + S11_t2(0lu, i);
			u1(0lu, i) = u1_t1(0lu, i) + u1_t2(0lu, i);
			u2(0lu, i) = u2_t1(0lu, i) + u2_t2(0lu, i);
			u3(0lu, i) = u3_t1(0lu, i) + u3_t2(0lu, i);

		}

		value.S331 = S331;
		value.S311 = S311;
		value.S321 = S321;
		value.S111 = S111;
		value.u1 = u1;
		value.u2 = u2;
		value.u3 = u3;

	}
	if (plotMode == 3) {

		double nTrg = nTrgX * nTrgZ;
		double nSweepTrg = nSweepTrgX * nTrgZ;

		cx_mat TR(1, 3 * nSweepTrgX * nTrgZ * nTrgY);
		for (int g = 0; g < nTrgY; g++) {

			double SrcIndex = g * nTrgX + (nTrgX + 1) / 2;

			// Coordinate of the central point source just 'Source_EqivR' distance below the interface
			double xSourceCent = IntrFcCoord_Btm->at(0, 0)(0, SrcIndex - 1);
			double ySourceCent = IntrFcCoord_Btm->at(0, 0)(1, SrcIndex - 1);
			double zSourceCent = IntrFcCoord_Btm->at(0, 0)(2, SrcIndex - 1);

			for (unsigned long h = 0; h < nTrgZ; h++) {
				for (unsigned long i = 0; i < nSweepTrgX; i++) {  // total no. of target points

					unsigned long Sw_TrgLoc = i + h * nSweepTrgX;
					unsigned long k         = Sw_TrgLoc + g * nSweepTrgX * nTrgZ;

					TR.col(3 * k + 0) = sweepTrgCoord(Sw_TrgLoc, 0) - xSourceCent;
					TR.col(3 * k + 1) = sweepTrgCoord(Sw_TrgLoc, 1) - ySourceCent;
					TR.col(3 * k + 2) = sweepTrgCoord(Sw_TrgLoc, 2) - zSourceCent;

					for (int i = 0; i < 3; i++) {
						std::complex<double> cmplx = TR(0, 3 * k + i);
						double real = abs(cmplx.real());
						if (real < tolerance) {
							TR(0, 3 * k + i) = std::complex<double>(0, 0);
						}
					}
				}
			}
		}

		std::unique_ptr<SolidGreen<double>> sg (new SolidGreen<double>(nSweepTrgX * nTrgZ, nTrgY, cache));
	    sg->solve(TR, C, cSphere, Solid_rho, freq, dTheta, dPhi, config);

		arma::cx_mat S33_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t1 (1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t1   (1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t1   (1, nTrg, arma::fill::zeros);

		complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgZ; h++) {
			for (int i = 0; i < nTrgX; i++) {				// for target points in X - Z plane
				for (int j = 0; j < nTrgY; j++) {
					for (int k = 0; k < nTrgX; k++) {		// for source points in X - Y plane

						double index  = k + j * nTrgX;
						double index1 = i + h * nSweepTrgX + nTrgX - k - 1;
						double index2 = i + h * nTrgX;

						//For Stress
						S33t = (A_1s(0, index)*sg->s33->at(index1, j, 0)) + (A_1s(1, index)*sg->s33->at(index1, j, 1)) + (A_1s(2, index)*sg->s33->at(index1, j, 2));
						S31t = (A_1s(0, index)*sg->s13->at(index1, j, 0)) + (A_1s(1, index)*sg->s13->at(index1, j, 1)) + (A_1s(2, index)*sg->s13->at(index1, j, 2));
						S32t = (A_1s(0, index)*sg->s23->at(index1, j, 0)) + (A_1s(1, index)*sg->s23->at(index1, j, 1)) + (A_1s(2, index)*sg->s23->at(index1, j, 2));
						S11t = (A_1s(0, index)*sg->s11->at(index1, j, 0)) + (A_1s(1, index)*sg->s11->at(index1, j, 1)) + (A_1s(2, index)*sg->s11->at(index1, j, 2));

						S33_t1(0, index2) = S33_t1(0, index2) + S33t;
						S31_t1(0, index2) = S31_t1(0, index2) + S31t;
						S32_t1(0, index2) = S32_t1(0, index2) + S32t;
						S11_t1(0, index2) = S11_t1(0, index2) + S11t;

						//For Displacement
						u1t = (A_1s(0, index)*sg->u1->at(index1, j, 0)) + (A_1s(1, index)*sg->u1->at(index1, j, 1)) + (A_1s(2, index)*sg->u1->at(index1, j, 2));
						u2t = (A_1s(0, index)*sg->u2->at(index1, j, 0)) + (A_1s(1, index)*sg->u2->at(index1, j, 1)) + (A_1s(2, index)*sg->u2->at(index1, j, 2));
						u3t = (A_1s(0, index)*sg->u3->at(index1, j, 0)) + (A_1s(1, index)*sg->u3->at(index1, j, 1)) + (A_1s(2, index)*sg->u3->at(index1, j, 2));

						u1_t1(0, index2) = u1_t1(0, index2) + u1t;
						u2_t1(0, index2) = u2_t1(0, index2) + u2t;
						u3_t1(0, index2) = u3_t1(0, index2) + u3t;

					}
				}
			}
		}

		// Calculation Starts for t2(Point Sources at Interface 2 - Top)
		// Implementing SISMAG to obtain Green's function for t2

		arma::cx_cube u1t2  (nSweepTrg, nTrgY, 3);
		arma::cx_cube u2t2  (nSweepTrg, nTrgY, 3);
		arma::cx_cube u3t2  (nSweepTrg, nTrgY, 3);
		arma::cx_cube S11t2 (nSweepTrg, nTrgY, 3);
		arma::cx_cube S33t2 (nSweepTrg, nTrgY, 3);
		arma::cx_cube S31t2 (nSweepTrg, nTrgY, 3);
		arma::cx_cube S32t2 (nSweepTrg, nTrgY, 3);

		for (unsigned long i = 0; i < nSweepTrg; i++) {
			for (unsigned long j = 0; j < nTrgY; j++) {

				// For Force along 1 direction
				u1t2.at(i, j, 0) = (sg->u1->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				u2t2.at(i, j, 0) = (sg->u2->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				u3t2.at(i, j, 0) = -(sg->u3->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				S11t2.at(i, j, 0) = (sg->s11->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				S33t2.at(i, j, 0) = (sg->s33->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				S31t2.at(i, j, 0) = -(sg->s13->at(nSweepTrg -1- i, nTrgY -1- j, 0));
				S32t2.at(i, j, 0) = -(sg->s23->at(nSweepTrg -1- i, nTrgY -1- j, 0));

				// For Force along 2 direction
				u1t2.at(i, j, 1) = (sg->u1->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				u2t2.at(i, j, 1) = (sg->u2->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				u3t2.at(i, j, 1) = -(sg->u3->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				S11t2.at(i, j, 1) = (sg->s11->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				S33t2.at(i, j, 1) = (sg->s33->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				S31t2.at(i, j, 1) = -(sg->s13->at(nSweepTrg -1- i, nTrgY -1- j, 1));
				S32t2.at(i, j, 1) = -(sg->s23->at(nSweepTrg -1- i, nTrgY -1- j, 1));

				// For Force along 3 direction
				u1t2.at(i, j, 2) = -(sg->u1->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				u2t2.at(i, j, 2) = -(sg->u2->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				u3t2.at(i, j, 2) = (sg->u3->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				S11t2.at(i, j, 2) = -(sg->s11->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				S33t2.at(i, j, 2) = -(sg->s33->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				S31t2.at(i, j, 2) = (sg->s13->at(nSweepTrg -1- i, nTrgY -1- j, 2));
				S32t2.at(i, j, 2) = (sg->s23->at(nSweepTrg -1- i, nTrgY -1- j, 2));

			}
		}

		arma::cx_mat S33_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S31_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S32_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat S11_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2_t2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3_t2(1, nTrg, arma::fill::zeros);
		
		//complex<double> S33t(0, 0), S31t(0, 0), S32t(0, 0), S11t(0, 0), u1t(0, 0), u2t(0, 0), u3t(0, 0);
		for (int h = 0; h < nTrgZ; h++) {
			for (int i = 0; i < nTrgX; i++) {				// for target points in X - Z plane
				for (int j = 0; j < nTrgY; j++) {
					for (int k = 0; k < nTrgX; k++) {		// for source points in X - Y plane

						double index = k + j * nTrgX;
						double index1 = i + h * nSweepTrgX + nTrgX - k - 1;
						double index2 = i + h * nTrgX;

						//For Stress
						S33t = (A_1s(0, index)*sg->s33->at(index1, j, 0)) + (A_1s(1, index)*sg->s33->at(index1, j, 1)) + (A_1s(2, index)*sg->s33->at(index1, j, 2));
						S31t = (A_1s(0, index)*sg->s13->at(index1, j, 0)) + (A_1s(1, index)*sg->s13->at(index1, j, 1)) + (A_1s(2, index)*sg->s13->at(index1, j, 2));
						S32t = (A_1s(0, index)*sg->s23->at(index1, j, 0)) + (A_1s(1, index)*sg->s23->at(index1, j, 1)) + (A_1s(2, index)*sg->s23->at(index1, j, 2));
						S11t = (A_1s(0, index)*sg->s11->at(index1, j, 0)) + (A_1s(1, index)*sg->s11->at(index1, j, 1)) + (A_1s(2, index)*sg->s11->at(index1, j, 2));

						S33_t2(0, index2) = S33_t2(0, index2) + S33t;
						S31_t2(0, index2) = S31_t2(0, index2) + S31t;
						S32_t2(0, index2) = S32_t2(0, index2) + S32t;
						S11_t2(0, index2) = S11_t2(0, index2) + S11t;

						//For Displacement
						u1t = (A_1s(0, index)*sg->u1->at(index1, j, 0)) + (A_1s(1, index)*sg->u1->at(index1, j, 1)) + (A_1s(2, index)*sg->u1->at(index1, j, 2));
						u2t = (A_1s(0, index)*sg->u2->at(index1, j, 0)) + (A_1s(1, index)*sg->u2->at(index1, j, 1)) + (A_1s(2, index)*sg->u2->at(index1, j, 2));
						u3t = (A_1s(0, index)*sg->u3->at(index1, j, 0)) + (A_1s(1, index)*sg->u3->at(index1, j, 1)) + (A_1s(2, index)*sg->u3->at(index1, j, 2));

						u1_t2(0, index2) = u1_t2(0, index2) + u1t;
						u2_t2(0, index2) = u2_t2(0, index2) + u2t;
						u3_t2(0, index2) = u3_t2(0, index2) + u3t;

					}
				}
			}
		}

		arma::cx_mat S331(1, nTrg, arma::fill::zeros);
		arma::cx_mat S311(1, nTrg, arma::fill::zeros);
		arma::cx_mat S321(1, nTrg, arma::fill::zeros);
		arma::cx_mat S111(1, nTrg, arma::fill::zeros);
		arma::cx_mat u1(1, nTrg, arma::fill::zeros);
		arma::cx_mat u2(1, nTrg, arma::fill::zeros);
		arma::cx_mat u3(1, nTrg, arma::fill::zeros);

		for (unsigned long i = 0; i < nTrg; i++) {

			S331(0lu, i) = S33_t1(0lu, i) + S33_t2(0lu, i);
			S311(0lu, i) = S31_t1(0lu, i) + S31_t2(0lu, i);
			S321(0lu, i) = S32_t1(0lu, i) + S32_t2(0lu, i);
			S111(0lu, i) = S11_t1(0lu, i) + S11_t2(0lu, i);
			u1(0lu, i) = u1_t1(0lu, i) + u1_t2(0lu, i);
			u2(0lu, i) = u2_t1(0lu, i) + u2_t2(0lu, i);
			u3(0lu, i) = u3_t1(0lu, i) + u3_t2(0lu, i);

		}

		value.S331 = S331;
		value.S311 = S311;
		value.S321 = S321;
		value.S111 = S111;
		value.u1 = u1;
		value.u2 = u2;
		value.u3 = u3;

	}

	printf("End CompWaveFieldSolid\n");

	return value;
}
