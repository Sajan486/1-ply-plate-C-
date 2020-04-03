#ifndef PRESSUREFLUIDMAT_H_
#define PRESSUREFLUIDMAT_H_

#include <stdio.h>
#include <math.h>
#include "armadillo"
#include <complex>
#include "Util.h"
#include "Config.h"

using namespace std;

class PressureFluidMat {
public:
	PressureFluidMat(unsigned long _nSrcTotal, unsigned long _nSrcTrans,
			vat::DiskCache* cache, vat::MemorySize maxBlockSize);

	~PressureFluidMat();

	void solve(const arma::mat& TransCoord_Cent,
		const arma::mat& TransCoord_Btm,
		const arma::mat& IntfaceCoord_Cent,
		const arma::mat& IntfaceCoord_Top,
		double     WaveNum_P,
		double     Fluid_rho,
		double     WaveVel_P,
		double     Rotation_Trans,
		int		   InterfaceIndex);
	
	void save(const std::string& = "");

	void replaceViews(vat::CxdMat* mssView, vat::CxdMat* qisView, vat::CxdMat* msiView, vat::CxdMat* qiiView);

	vat::CxdMat& getMSS();
	vat::CxdMat& getMsi();
	vat::CxdMat& getQis();
	vat::CxdMat& getQii();

	vat::CxdMat& getMis();
	vat::CxdMat& getMii();
	vat::CxdMat& getQss();
	vat::CxdMat& getQsi();


private:
	unsigned long nSrcTotal;
	unsigned long nSrcTrans;

	vat::CxdMat* Mss;
	vat::CxdMat* Msi;
	vat::CxdMat* Qis;
	vat::CxdMat* Qii;

	std::unique_ptr<vat::CxdMat> Mis;
	std::unique_ptr<vat::CxdMat> Mii;
	std::unique_ptr<vat::CxdMat> Qss;
	std::unique_ptr<vat::CxdMat> Qsi;
};

class LinePressureFluidMat {
public:
	LinePressureFluidMat(unsigned long _nSrcTotal, unsigned long _nSrcTrans,
			unsigned long _NumLinePt, vat::DiskCache* cache, vat::MemorySize maxBlockSize);
	~LinePressureFluidMat();

	void solve(const arma::mat& Line,
			const arma::mat& TransCoord_Source,
			const arma::mat& IntfaceCoord_Source,
			double     WaveNum_P,
			double     Fluid_rho,
			double     WaveVel_P,
			double     Rotation_Trans,
			int		   InterfaceIndex);

	void save(const std::string& path = "");

	arma::cx_mat& getMls();
	arma::cx_mat& getMli();
	arma::cx_mat& getQls();
	arma::cx_mat& getQli();

private:
	unsigned long nSrcTotal;
	unsigned long nSrcTrans;
	unsigned long NumLinePt;

	arma::cx_mat* Mls;
	arma::cx_mat* Mli;
	arma::cx_mat* Qls;
	arma::cx_mat* Qli;
};
#endif
