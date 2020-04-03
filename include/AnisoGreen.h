#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "Util.h"
#include "SolidGreen.h"
#include "armadillo"

#ifndef ANISOGREEN_H_
#define ANISOGREEN_H_

template<typename F>
class AnisoGreen {
public:
	AnisoGreen(unsigned long nSrcTot, vat::DiskCache* cache, vat::MemorySize blockSize);
	~AnisoGreen();

	// Granular option
	void solve(const arma::mat*          Sw_IntrFcCoord_Cent,
			   const arma::mat*          CoordCentSourcePt,
		       ChristofelSphere& 	 	 cSphere,
		       const arma::mat*          C,
		       F                         nSrcX,
		       F                         nSrcY,
		       F                         Solid_rho,
		       F                         freq,
		       F                         dTheta,
		       F                         dPhi,
		       Config&					 config,
		       vat::DiskCache*			 cache);

	vat::CxMat<F>** cDS1;
	vat::CxMat<F>** cDS2;
	vat::CxMat<F>** cDS3;
	vat::CxMat<F>** cS11;
	vat::CxMat<F>** cS22;
	vat::CxMat<F>** cS33;
	vat::CxMat<F>** cS12;
	vat::CxMat<F>** cS31;
	vat::CxMat<F>** cS32;
	vat::CxMat<F>*** cData;

};
template class AnisoGreen<float>;
template class AnisoGreen<double>;

#endif
