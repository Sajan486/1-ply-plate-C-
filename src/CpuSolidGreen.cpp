#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <complex>
#include "armadillo"
#include "SolidGreen.h"

using namespace std;
using namespace arma;

template<typename F>
void cpuSolidGreen(SolidGreen<F>*            output,
				   arma::cx_mat&       		 aTR,
				   const arma::mat&          C,
				   ChristofelSphere&   		 cSphere,
				   double                    Fluid_rho,
				   double                    freq,
				   double                    dTheta,
				   double                    dPhi,
				   double                    dispSign)
{
	printf("Beginning Solid Green function\n");

	unsigned long nTestPt = cSphere.getSphere().getnPts();

	//aTR.print();

	std::clock_t start;
	start = std::clock();
	double duration;

	long nTargetPts = output->nX * output->nY;

	std::complex<double> cxZero(0.0, 0.0);

	long n;

	for (int l = 0; l < output->nY; l++) {
		for (int m = 0; m < output->nX; m++) {

			n = m + l * output->nX;
			mat Tr(1, 3);
			for (int i = 0; i < 3; i++) Tr(0, i) = aTR(0, 3 * n + i).real();
	//			Tr.print();

			complex<double> img(0, 1);
    		complex<double> gam = 0 ;
    		double IntfaceArea = 0, mag = 0;
    		long nTestPtSq = nTestPt * nTestPt;

			cx_mat uu1(3, 3), uu2(3, 3), U_Dw1(3, 3), U_Dw2(3, 3), U_Dw3(3, 3), DGF(3,3), u1(1,3), u2(1,3), u3(1,3);
			cx_mat eps1(6, 1), eps2(6, 1), eps3(6, 1), sig1(6, 1), sig2(6, 1), sig3(6, 1), S11(1,3), S22(1, 3), S33(1, 3), S12(1, 3), S13(1, 3), S23(1, 3);
			cx_mat vec(3, 3), Pij(3, 3), soln1(3,3), soln2(3, 3), DispDw1(3, 3), DispDw2(3, 3), DispDw3(3, 3), DispD_2(3, 3);


    		mat v(1, 3), F_bar(3, 3);

			F_bar	<< 1 << 0 << 0 << endr
					<< 0 << 1 << 0 << endr
					<< 0 << 0 << 1 << endr;

			int zaa = 0;
			int zbb = 0;
			int zcc = 0;
			for (int mode = 0; mode < 3; mode++) {
				for (int i = 0; i < nTestPt; i++) {
					for (int j = 0; j < nTestPt; j++) {


						v(0, 0) = cos(cSphere.getSphere().getTheta()(i, j))*cos(cSphere.getSphere().getPhi()(i, j));
						v(0, 1) = sin(cSphere.getSphere().getTheta()(i, j))*cos(cSphere.getSphere().getPhi()(i, j));
						v(0, 2) = sin(cSphere.getSphere().getPhi()(i, j));

						vec = cSphere.getFI()(i, j);
						Pij = vec.col(mode) * trans(vec.col(mode));
						for (int a = 0;a < 3;a++) {
							for (int b = 0;b < 3;b++) {
								Pij(a, b) = round(real(Pij(a, b))*1e8)/1e8;
							}
						}

						gam = sqrt(cSphere.getCV()(i, j)(0, mode)); //CV(i,j) is a diagonal matrix in C++
						IntfaceArea = dot(Tr, v);
						mag = sqrt(Tr(0, 0)*Tr(0, 0) + Tr(0, 1)*Tr(0, 1) + Tr(0, 2)*Tr(0, 2));


						if (IntfaceArea > 0) {
							zbb++;
							soln1 = (pow(gam, -3) * Pij*exp(img*freq*IntfaceArea / gam)*(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j))));					// for displacement
							soln2.fill(cxZero);

							DispDw1 = (((complex<double>)1 / pow(gam, 4))*v(0, 0)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));		// disp.derivative freq.R.t.x1
							DispDw2 = (((complex<double>)1 / pow(gam, 4))*v(0, 1)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));		//disp.derivative freq.R.t.x2
							DispDw3 = (((complex<double>)1 / pow(gam, 4))*v(0, 2)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));		// disp.derivative freq.R.t.x3
							DispD_2.fill(cxZero);
						}
						else if (IntfaceArea == 0) {
							zaa++;

							soln1 = (pow(gam,-3) * Pij*exp(img*freq*IntfaceArea / gam)*(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j))));
							soln2 = (pow(gam,-2) * Pij)*dPhi;

							DispDw1 = (((complex<double>)1 / pow(gam, 4))*v(0, 0)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));//(((complex<double>)1 / pow(gam, 4))*v(0, 0)*Pij*exp(img*freq*IntfaceArea) / gam)*(dTheta*dPhi*cos(Phi(i, j)));		// disp.derivative freq.R.t.x1
							DispDw2 = (((complex<double>)1 / pow(gam, 4))*v(0, 1)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));		// disp.derivative freq.R.t.x2
							DispDw3 = (((complex<double>)1 / pow(gam, 4))*v(0, 2)*Pij*exp(img*freq*IntfaceArea) / gam)*
									(dTheta*dPhi*cos(cSphere.getSphere().getPhi()(i, j)));		// disp.derivative freq.R.t.x3
							DispD_2 = (1.0 / pow(gam, 2)) * Pij * dPhi;
						}
						else {
							zcc++;
							soln1.fill(cxZero);
    		                soln2.fill(cxZero);

							DispDw1.fill(cxZero);
							DispDw2.fill(cxZero);
							DispDw3.fill(cxZero);
							DispD_2.fill(cxZero);
						}

						uu1 = uu1 + ((img * freq) / (8 * (pow(M_PI, 2))*Fluid_rho)) * soln1 * F_bar;		    // First part of disp.green's function
    		                                                                                                    // Please note it is nor necessary to have F_bar as multyiplying
    		                                                                                                    // a Matrix with an Identity matrix result the same matrix. Banerjee 07/18/2016

						uu2 = uu2 + (1 / (8 * (pow(M_PI, 2)) * Fluid_rho * mag)) * soln2 * F_bar;				// Second part of disp.green's function

						U_Dw1 += (DispDw1*(-(pow(freq, 2)) / (8 * (pow(M_PI, 2))*Fluid_rho)));
						U_Dw2 += (DispDw2*(-(pow(freq, 2)) / (8 * (pow(M_PI, 2))*Fluid_rho)));
						U_Dw3 += (DispDw3*(-(pow(freq, 2)) / (8 * (pow(M_PI, 2))*Fluid_rho)));

						// if (testSwitch) DispD_2.print();
						U_Dw1 -= (DispD_2 * (Tr(0, 0) / (8 * pow(M_PI, 2) * Fluid_rho * pow(mag, 3))));
						U_Dw2 -= (DispD_2 * (Tr(0, 1) / (8 * pow(M_PI, 2) * Fluid_rho * pow(mag, 3))));
						U_Dw3 -= (DispD_2 * (Tr(0, 2) / (8 * pow(M_PI, 2) * Fluid_rho * pow(mag, 3))));
					}
				}
			}

			//uu2.print();


			// Total Displacement Green's Function Due to all 3 wave modes.
			DGF = uu1 + uu2;

			//For Displacements Green's function Allocation
			u1 = DGF.row(0);
			u2 = DGF.row(1);
			u3 = DGF.row(2);
			// For Stress Green's Function

			// Strains
			eps1 << U_Dw1(0, 0) << endr << U_Dw2(1, 0) << endr << U_Dw3(2, 0) << endr << 0.5*(U_Dw3(1, 0) + U_Dw2(2, 0)) << endr << 0.5*(U_Dw1(2, 0) + U_Dw3(0, 0)) << endr << 0.5*(U_Dw1(1, 0) + U_Dw2(0, 0)) << endr;
			eps2 << U_Dw1(0, 1) << endr << U_Dw2(1, 1) << endr << U_Dw3(2, 1) << endr << 0.5*(U_Dw3(1, 1) + U_Dw2(2, 1)) << endr << 0.5*(U_Dw1(2, 1) + U_Dw3(0, 1)) << endr << 0.5*(U_Dw1(1, 1) + U_Dw2(0, 1)) << endr;
			eps3 << U_Dw1(0, 2) << endr << U_Dw2(1, 2) << endr << U_Dw3(2, 2) << endr << 0.5*(U_Dw3(1, 2) + U_Dw2(2, 2)) << endr << 0.5*(U_Dw1(2, 2) + U_Dw3(0, 2)) << endr << 0.5*(U_Dw1(1, 2) + U_Dw2(0, 2)) << endr;

			// Stress Using Constitutive Eqn
			sig1 = C*eps1;      // FOR force along 1 - direction
			sig2 = C*eps2;      // FOR force along 2 - direction
			sig3 = C*eps3;      // FOR force along 3 - direction

			// Stress Green's function Allocation
			S11 << sig1(0, 0) << sig2(0, 0)<< sig3(0, 0) << endr;
			S22 << sig1(1, 0) << sig2(1, 0)<< sig3(1, 0) << endr;
			S33 << sig1(2, 0) << sig2(2, 0)<< sig3(2, 0) << endr;
			S12 << sig1(5, 0) << sig2(5, 0)<< sig3(5, 0) << endr;
			S13 << sig1(4, 0) << sig2(4, 0)<< sig3(4, 0) << endr;
			S23 << sig1(3, 0) << sig2(3, 0)<< sig3(3, 0) << endr;

			for (int i = 0; i < 3; i++) {
				output->u1->at(m, l, i) = dispSign * u1(0, i);
				output->u2->at(m, l, i) = dispSign * u2(0, i);
				output->u3->at(m, l, i) = dispSign * u3(0, i);

				output->s11->at(m, l, i) = S11(0, i);
				output->s22->at(m, l, i) = S22(0, i);
				output->s33->at(m, l, i) = S33(0, i);
				output->s12->at(m, l, i) = S12(0, i);
				output->s13->at(m, l, i) = S13(0, i);
				output->s23->at(m, l, i) = S23(0, i);
			}

			bool printUpdates = true;
			bool lastIter = (n == nTargetPts - 1);
			long itersPerPrint = 3000;
			if (((n != 0 && n % itersPerPrint == 0) || lastIter) && printUpdates) {
				duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
				double iterDuration = (duration) / ((double) n + 1);
			    long remainingIters = nTargetPts - n - 1;

			    double remainingTime = remainingIters * iterDuration;
			    double remainingTimeHr = floor(remainingTime / 3600.0);
			    double remainingTimeMin = floor(remainingTime / 60.0) - 60.0 * remainingTimeHr;
			    double remainingTimeSec = remainingTime - 3600.0 * remainingTimeHr - 60.0 * remainingTimeMin;
			    double percentComplete = 100 * (double) n / nTargetPts;
			    if (!lastIter) {
			    	if (remainingTimeHr > 0) {
			    		printf("Remaining Time: %.0f hrs, %.0f min (%.2lf%% complete)\n", remainingTimeHr, remainingTimeMin, percentComplete);
			    	} else {
			    		printf("Remaining Time: %.0f min, %.0f sec (%.2lf%% complete)\n", remainingTimeMin, remainingTimeSec, percentComplete);
			    	}
			    } else {
			    	printf("Remaining Time: %.0f min, %.0f sec (%.2lf%% complete)\n", 0.0, 0.0, 100.0);
			    }
			}
		}
	}
}

template void cpuSolidGreen<float> (SolidGreen<float>*,   arma::cx_mat&, const arma::mat&, ChristofelSphere&, double, double, double, double, double);
template void cpuSolidGreen<double>(SolidGreen<double>*,  arma::cx_mat&, const arma::mat&, ChristofelSphere&, double, double, double, double, double);
