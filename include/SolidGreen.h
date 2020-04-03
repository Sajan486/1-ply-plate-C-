#ifndef SOLIDGREEN_H_
#define SOLIDGREEN_H_

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <tuple>
#include <sys/time.h>
#include <vector>
#include <map>
#include <math.h>
#include <complex>
#include <mutex>

#include "armadillo"

#include "ChristofelSphere.h"
#include "Matrix.h"
#include "cdm.h"

template<typename F>
class SolidGreen;

/*
/*
 * @param output: Pointer to a class containing all output matrices
 * @param aTR:    Arma matrix containing all target points
 * @param C:      Material properties matrix
 * @param Theta:  Lateral test point angles
 * @param Phi:    Vertical test point angles
 *
 * Function is templated to accomodate single or double floating point precision
 */
template<typename F>
void cuSolidGreen(SolidGreen<double>*       sg,
				  arma::cx_mat&       aTR,
				  const arma::mat&          C,
				  ChristofelSphere&   		cSphere,
				  F                         Fluid_rho,
				  F                         freq,
				  F                         dTheta,
				  F                         dPhi,
				  F                         dispSign,
				  Config&					config);

/*
 * @param output: Pointer to a class containing all output matrices
 * @param aTR:    Arma matrix containing all target points
 * @param C:      Material properties matrix
 * @param Theta:  Lateral test point angles
 * @param Phi:    Vertical test point angles
 *
 * Function is templated to accomodate single or double floating point precision
 */
template<typename F>
void cpuSolidGreen(SolidGreen<F>*           sg,
				  arma::cx_mat&       aTR,
				  const arma::mat&          C,
				  ChristofelSphere&   		cSphere,
				  double                    Fluid_rho,
				  double                    freq,
				  double                    dTheta,
				  double                    dPhi,
				  double                    dispSign);

template<typename F>
class SolidGreen {
public:

	SolidGreen(long _nX, long _nY, vat::DiskCache* cache) {
		nX = _nX;
		nY = _nY;

		data = new arma::cx_cube*[9];
		for (int i = 0; i < 9; i++) {
			data[i] = new arma::cx_cube(nX, nY, 3);
		}
		u1  = data[0]; u2  = data[1]; u3  = data[2];
		s11 = data[3]; s22 = data[4]; s33 = data[5];
		s23 = data[6]; s13 = data[7]; s12 = data[8];
	}

	virtual ~SolidGreen() {
		for (int i = 0; i < 9; i++) {
			data[i]->reset();
			delete data[i];
		}
		delete[] data;
	}

	void solve(arma::cx_mat&              aTR,
			   const arma::mat&           C,
			   ChristofelSphere&	  	  cSphere,
		       double                     Fluid_rho,
		       double                     freq,
	           double                     dTheta,
	           double                     dPhi,
	           Config&					  config,
	           double                     dispSign = 1)
	{
		if (config.enableGPU) {
			/*
			 * Single Precision for the solid green function is forced here. Precision
			 * should be uniform throughout the program, but integrating throughout the
			 * codebase created unexpected bugs. The results for single precision;
			 * however are almost exactly the same as double, but they are processed
			 * >40% faster
			 */
			if (config.doublePrecision) {
				cuSolidGreen<double>(this, aTR, C, cSphere, Fluid_rho, freq, dTheta, dPhi, dispSign, config);
			} else {
				cuSolidGreen<float>(this, aTR, C, cSphere, (float)Fluid_rho, (float)freq, (float)dTheta, (float)dPhi, (float)dispSign, config);
			}
		} else {
			cpuSolidGreen<F>(this, aTR, C, cSphere, Fluid_rho, freq, dTheta, dPhi, dispSign);
		}
	}

	void validate();

	arma::cx_cube* u1;
	arma::cx_cube* u2;
	arma::cx_cube* u3;
	arma::cx_cube* s11;
	arma::cx_cube* s22;
	arma::cx_cube* s33;
	arma::cx_cube* s12;
	arma::cx_cube* s13;
	arma::cx_cube* s23;

	long nX;
	long nY;

	arma::cx_cube** data;
};

// options for precision level
//template class SolidGreen<float>;
template class SolidGreen<double>;

#endif
