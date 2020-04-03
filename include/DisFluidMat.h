#ifndef DISFLUIDMAT_H_
#define DISFLUIDMAT_H_

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "armadillo"
#include "Config.h"
#include <chrono>
#include "Matrix.h"
#include "Util.h"

class DisFluidMat {
public:
	enum DisMat {
		DF3ssView, DF3isView, DF3siView, DF3iiView
	};

	DisFluidMat(unsigned long _nSrcTrans, unsigned long _nSrcTotal, vat::DiskCache* cache,
			vat::MemorySize maxBlockSize);
	~DisFluidMat();

	void solve(const arma::mat& TransCoord_Cent,
			   const arma::mat& TransCoord_Btm,
			   const arma::mat& IntrFcCoord_Cent,
			   const arma::mat& IntrFcCoord_Top,
			   double WaveNum_P,
			   double freq,
			   double Fluid_rho);

	void replaceView(DisMat mat, vat::CxdMat* disView);

	void save(const std::string& path = "");

	vat::CxdMat& getDF3ss();
	vat::CxdMat& getDF3is();
	vat::CxdMat& getDF3si();
	vat::CxdMat& getDF3ii();

	unsigned long getnSrcTot();
	unsigned long getnSrcTrans();


private:
	vat::CxdMat* DF3ss;
	vat::CxdMat* DF3is;
	vat::CxdMat* DF3si;
	vat::CxdMat* DF3ii;

	unsigned long nSrcTot;
	unsigned long nSrcTrans;
};

class LineDisFluidMat {
public:
	LineDisFluidMat(unsigned long _nSrcTrans, unsigned long _nSrcTotal, unsigned long _NumLinePt);
	~LineDisFluidMat();

	void solve(const arma::mat& Line,
		const arma::mat& TransCoord_Source,
		const arma::mat& IntrFcCoord_Source,
		double WaveNum_P,
		double freq,
		double Fluid_rho);

	void save(const std::string& path = "");

	arma::cx_mat& getDF3ls();
	arma::cx_mat& getDF3li();

	unsigned long getnSrcTot();
	unsigned long getnSrcTrans();
	unsigned long getNumLinePt();

private:
	arma::cx_mat* DF3ls;
	arma::cx_mat* DF3li;

	unsigned long nSrcTot;
	unsigned long nSrcTrans;
	unsigned long NumLinePt;
};
#endif
