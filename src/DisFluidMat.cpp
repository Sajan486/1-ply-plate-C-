#include "DisFluidMat.h"

DisFluidMat::DisFluidMat(unsigned long _nSrcTrans, unsigned long _nSrcTotal, vat::DiskCache* cache,
		vat::MemorySize maxBlockSize) {
	nSrcTot		= _nSrcTotal;
	nSrcTrans	= _nSrcTrans;

	DF3ss = new vat::CxdMat(cache, nSrcTrans, nSrcTrans, maxBlockSize);
	DF3is = new vat::CxdMat(cache, nSrcTot, nSrcTrans, maxBlockSize);
	DF3si = new vat::CxdMat(cache, nSrcTrans, nSrcTot, maxBlockSize);
	DF3ii = new vat::CxdMat(cache, nSrcTot, nSrcTot, maxBlockSize);
}

DisFluidMat::~DisFluidMat() {
	delete DF3ss;
	delete DF3ii;
	delete DF3is;
	delete DF3si;
}

void DisFluidMat::save(const std::string& path) {
	toArma(DF3ss).save(path + "DF3ss", arma::csv_ascii);
	toArma(DF3ii).save(path + "DF3ii", arma::csv_ascii);
	toArma(DF3is).save(path + "DF3is", arma::csv_ascii);
	toArma(DF3si).save(path + "DF3si", arma::csv_ascii);
}

void DisFluidMat::solve(const arma::mat& TransCoord_Cent,
	const arma::mat& TransCoord_Btm,
	const arma::mat& IntrFcCoord_Cent,
	const arma::mat& IntrFcCoord_Top,
	double WaveNum_P,
	double freq,
	double Fluid_rho) {

	int indice;
	int i;

	std::complex<double> img(0, 1);
	double cons = 1 / (Fluid_rho*(pow(2 * M_PI*freq, 2)));
	double CoordCentSourcePt_x = 0, CoordCentSourcePt_y = 0, CoordCentSourcePt_z = 0;

	// DF3ss
	double R = 0, R3 = 0;
	for (indice = 0; indice < nSrcTrans; indice++) {

		CoordCentSourcePt_x = TransCoord_Btm(0, indice);
		CoordCentSourcePt_y = TransCoord_Btm(1, indice);
		CoordCentSourcePt_z = TransCoord_Btm(2, indice);

		arma::cx_vec colDF3ss(nSrcTrans);

		// total no. of target point
		for (i = 0; i < nSrcTrans; i++) {

			R = sqrt(pow(TransCoord_Cent(0, i) - CoordCentSourcePt_x, 2) + pow(TransCoord_Cent(1, i) - CoordCentSourcePt_y, 2) + pow(TransCoord_Cent(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((TransCoord_Cent(2, i) - CoordCentSourcePt_z) / R);
			colDF3ss(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
		}
		std::complex<double>* memCol = colDF3ss.memptr();
		DF3ss->col(memCol, indice);
	//		delete[] memCol;
	}

	//  DF3is
	for (indice = 0; indice < nSrcTrans; indice++) {

		CoordCentSourcePt_x = TransCoord_Btm(0, indice);
		CoordCentSourcePt_y = TransCoord_Btm(1, indice);
		CoordCentSourcePt_z = TransCoord_Btm(2, indice);

		arma::cx_vec colDF3is(nSrcTot);

		// total no. of target point
		for (i = 0; i < nSrcTot; i++) {

			R = sqrt(pow(IntrFcCoord_Cent(0, i) - CoordCentSourcePt_x, 2) + pow(IntrFcCoord_Cent(1, i) - CoordCentSourcePt_y, 2) + pow(IntrFcCoord_Cent(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((IntrFcCoord_Cent(2, i) - CoordCentSourcePt_z) / R);
			colDF3is(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
		}
		std::complex<double>* memCol = colDF3is.memptr();
		DF3is->col(memCol, indice);
		//delete[] memCol;
	}

	// DF3si
	for (indice = 0; indice < nSrcTot; indice++) {

		CoordCentSourcePt_x = IntrFcCoord_Top(0, indice);
		CoordCentSourcePt_y = IntrFcCoord_Top(1, indice);
		CoordCentSourcePt_z = IntrFcCoord_Top(2, indice);

		arma::cx_vec colDF3si(nSrcTrans);

		// total no. of target point
		for (i = 0; i < nSrcTrans; i++) {

			R = sqrt(pow(TransCoord_Cent(0, i) - CoordCentSourcePt_x, 2) + pow(TransCoord_Cent(1, i) - CoordCentSourcePt_y, 2) + pow(TransCoord_Cent(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((TransCoord_Cent(2, i) - CoordCentSourcePt_z) / R);
			colDF3si(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
		}
		std::complex<double>* memCol = colDF3si.memptr();
		DF3si->col(memCol, indice);
		//delete memCol;
	}

	// DF3ii
	for (indice = 0; indice < nSrcTot; indice++) {

		CoordCentSourcePt_x = IntrFcCoord_Top(0, indice);
		CoordCentSourcePt_y = IntrFcCoord_Top(1, indice);
		CoordCentSourcePt_z = IntrFcCoord_Top(2, indice);

		arma::cx_vec colDF3ii(nSrcTot);

		// total no. of target point
		for (i = 0; i < nSrcTot; i++) {

			R = sqrt(pow(IntrFcCoord_Cent(0, i) - CoordCentSourcePt_x, 2) + pow(IntrFcCoord_Cent(1, i) - CoordCentSourcePt_y, 2) + pow(IntrFcCoord_Cent(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((IntrFcCoord_Cent(2, i) - CoordCentSourcePt_z) / R);
			colDF3ii(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
		}
		std::complex<double>* memCol = colDF3ii.memptr();
		DF3ii->col(memCol, indice);
	//	delete[] memCol;
	}
}

void DisFluidMat::replaceView(DisMat mat, vat::CxdMat* disView) {
	if (mat == DisMat::DF3iiView) {
		delete DF3ii;
		DF3ii = disView;
	}
	else if (mat == DisMat::DF3isView) {
		delete DF3is;
		DF3is = disView;
	}
	else if (mat == DisMat::DF3siView) {
		delete DF3si;
		DF3si = disView;
	}
	else if (mat == DisMat::DF3ssView) {
		delete DF3ss;
		DF3ss = disView;
	}
}

vat::CxdMat& DisFluidMat::getDF3ss() {
	return *DF3ss;
}

vat::CxdMat& DisFluidMat::getDF3is() {
	return *DF3is;
}

vat::CxdMat& DisFluidMat::getDF3si() {
	return *DF3si;
}

vat::CxdMat& DisFluidMat::getDF3ii() {
	return *DF3ii;
}

unsigned long DisFluidMat::getnSrcTot() {
	return nSrcTot;
}

unsigned long DisFluidMat::getnSrcTrans() {
	return nSrcTrans;
}


LineDisFluidMat::LineDisFluidMat(unsigned long _nSrcTrans, unsigned long _nSrcTotal, unsigned long _NumLinePt) {
	nSrcTot = _nSrcTotal;
	nSrcTrans = _nSrcTrans;
	NumLinePt = _NumLinePt;

	DF3ls = (new arma::cx_mat(NumLinePt, nSrcTrans));
	DF3li = new arma::cx_mat(NumLinePt, nSrcTot);
}

LineDisFluidMat::~LineDisFluidMat() {
	delete DF3ls;
	delete DF3li;
}

void LineDisFluidMat::solve(const arma::mat& Line,
	const arma::mat& TransCoord_Source,
	const arma::mat& IntrFcCoord_Source,
	double WaveNum_P,
	double freq,
	double Fluid_rho) {

	int indice;
	int i;

	std::complex<double> img(0, 1);
	double cons = 1 / (Fluid_rho*(pow(2 * M_PI*freq, 2)));
	double CoordCentSourcePt_x = 0, CoordCentSourcePt_y = 0, CoordCentSourcePt_z = 0;

	// DF3ss
	double R = 0, R3 = 0;
	for (indice = 0; indice < nSrcTrans; indice++) {

		CoordCentSourcePt_x = TransCoord_Source(0, indice);
		CoordCentSourcePt_y = TransCoord_Source(1, indice);
		CoordCentSourcePt_z = TransCoord_Source(2, indice);

		arma::cx_vec colDF3ss(NumLinePt);

		// total no. of target point
		for (i = 0; i < NumLinePt; i++) {

			R = sqrt(pow(Line(0, i) - CoordCentSourcePt_x, 2) + pow(Line(1, i) - CoordCentSourcePt_y, 2) + pow(Line(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((Line(2, i) - CoordCentSourcePt_z) / R);
			colDF3ss(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
			DF3ls->at(i, indice) = colDF3ss(i);
			//lineoutput->DF3ls->element(colDF3ss(i),i, indice);

		}
	//	lineoutput->DF3ls->col(colDF3ss.memptr(), indice);
		//std::cout<<indice<<std::endl;
	}

	std::cout<<"done with DF3ls\n"<<std::endl;
	// DF3si
	for (indice = 0; indice < nSrcTot; indice++) {

		CoordCentSourcePt_x = IntrFcCoord_Source(0, indice);
		CoordCentSourcePt_y = IntrFcCoord_Source(1, indice);
		CoordCentSourcePt_z = IntrFcCoord_Source(2, indice);

		arma::cx_vec colDF3si(NumLinePt);

		// total no. of target point
		for (i = 0; i < NumLinePt; i++) {

			R = sqrt(pow(Line(0, i) - CoordCentSourcePt_x, 2) + pow(Line(1, i) - CoordCentSourcePt_y, 2) + pow(Line(2, i) - CoordCentSourcePt_z, 2));
			R3 = ((Line(2, i) - CoordCentSourcePt_z) / R);
			colDF3si(i) = cons * ((img*WaveNum_P*R3*(exp(img*WaveNum_P*R)) / R) - (R3*(exp(img*WaveNum_P*R)) / pow(R, 2)));
			DF3li->at(i, indice) = colDF3si(i);
			//lineoutput->DF3li->element(colDF3si(i),i, indice);

		}
		//lineoutput->DF3li->col(colDF3si.memptr(), indice);
		//std::cout<<indice<<std::endl;
	}
}

void LineDisFluidMat::save(const std::string& path) {
	DF3li->save(path + "DF3li", arma::csv_ascii);
	DF3ls->save(path + "DF3ls", arma::csv_ascii);
}

arma::cx_mat& LineDisFluidMat::getDF3ls() {
	return *DF3ls;
}

arma::cx_mat& LineDisFluidMat::getDF3li() {
	return *DF3li;
}

unsigned long LineDisFluidMat::getnSrcTot() {
	return nSrcTot;
}

unsigned long LineDisFluidMat::getnSrcTrans() {
	return nSrcTrans;
}
