#include "PressureFluidMat.h"

/* DEPRECATED
PressureFluidMat::PressureFluidMat() {
	init((unsigned long)config::geom::NumSourcePt_IntrFc_Total, (unsigned long)config::transducer::NumSourcePt_Trans);
}
*/

PressureFluidMat::PressureFluidMat(unsigned long _nSrcTotal, unsigned long _nSrcTrans,
		vat::DiskCache* cache, vat::MemorySize maxBlockSize) {
	nSrcTotal = _nSrcTotal;
	nSrcTrans = _nSrcTrans;

	Mss = new vat::CxdMat(cache, nSrcTrans, nSrcTrans, maxBlockSize);
	Msi = new vat::CxdMat(cache, nSrcTrans, nSrcTotal, maxBlockSize);
	Qis = new vat::CxdMat(cache, nSrcTotal, nSrcTrans, maxBlockSize);
	Qii = new vat::CxdMat(cache, nSrcTotal, nSrcTotal, maxBlockSize);

	Mis = std::unique_ptr<vat::CxdMat> (new vat::CxdMat(cache, nSrcTotal, nSrcTrans, maxBlockSize));
	Mii = std::unique_ptr<vat::CxdMat> (new vat::CxdMat(cache, nSrcTotal, nSrcTotal, maxBlockSize));
	Qss = std::unique_ptr<vat::CxdMat> (new vat::CxdMat(cache, nSrcTrans, nSrcTrans, maxBlockSize));
	Qsi = std::unique_ptr<vat::CxdMat> (new vat::CxdMat(cache, nSrcTrans, nSrcTotal, maxBlockSize));
}

PressureFluidMat::~PressureFluidMat() {
	delete Mss;
	delete Qis;
	delete Msi;
	delete Qii;

	Mis.reset();
	Mii.reset();
	Qss.reset();
	Qsi.reset();
}

void PressureFluidMat::replaceViews(vat::CxdMat* mssView, vat::CxdMat* qisView, vat::CxdMat* msiView, vat::CxdMat* qiiView) {
	delete Mss; delete Qis; delete Msi; delete Qii;
	Mss = mssView; Qis = qisView; Msi = msiView; Qii = qiiView;
}

void PressureFluidMat::solve (const arma::mat& TransCoord_Cent,
	const arma::mat& TransCoord_Btm,
	const arma::mat& IntfaceCoord_Cent,
	const arma::mat& IntfaceCoord_Top,
	double     WaveNum_P,
	double     Fluid_rho,
	double     WaveVel_P,
	double     Rotation_Trans,
	int		   InterfaceIndex) {

	complex<double> img(0, 1);
	Rotation_Trans = Rotation_Trans * M_PI / 180;

	unsigned long is, jt;

	// Calculation of MSS
	arma::mat R(nSrcTrans, nSrcTrans);
	arma::mat x(nSrcTrans, nSrcTrans);
	arma::mat z(nSrcTrans, nSrcTrans);
	double CoordCentSourcePt_x, CoordCentSourcePt_y, CoordCentSourcePt_z;

	for (int indice = 0; indice <nSrcTrans; indice++) {
		CoordCentSourcePt_x = TransCoord_Btm(0, indice);
		CoordCentSourcePt_y = TransCoord_Btm(1, indice);
		CoordCentSourcePt_z = TransCoord_Btm(2, indice);

		for (int i = 0; i < nSrcTrans; i++) {
			R(indice, i) = sqrt(pow((TransCoord_Cent(0, i) - CoordCentSourcePt_x), 2) + pow((TransCoord_Cent(1, i) - CoordCentSourcePt_y), 2) + pow((TransCoord_Cent(2, i) - CoordCentSourcePt_z), 2));
			x(indice, i) = ((TransCoord_Cent(0, i) - CoordCentSourcePt_x));
			z(indice, i) = ((TransCoord_Cent(2, i) - CoordCentSourcePt_z));
		}
	}
	complex<double> MxSS(0, 0), MzSS(0, 0);
	for (is = 0; is < nSrcTrans; is++) {
		arma::cx_vec rowMSS(nSrcTrans);
		arma::cx_vec rowQSS(nSrcTrans);

		for (jt = 0; jt < nSrcTrans; jt++) {
			MxSS = ((x(is, jt)*exp(img*WaveNum_P*R(is, jt)))*(img*WaveNum_P - (1 / R(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*pow(R(is, jt), 2));
			MzSS = ((z(is, jt) * exp(img*WaveNum_P*R(is, jt)))*(img*WaveNum_P - (1 / R(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R(is, jt), 2)));

			if (InterfaceIndex == 0) {
				rowMSS(jt) = -MxSS * sin(Rotation_Trans) + MzSS * cos(Rotation_Trans);
			}
			else if (InterfaceIndex == 1) {
				rowMSS(jt) = -MxSS * sin(Rotation_Trans) - MzSS * cos(Rotation_Trans);
			}
			else {
				throw std::runtime_error("interface index must be 0 or 1.");
			}

			rowQSS(jt) = (exp(img*WaveNum_P*R(is, jt))) / (R(is, jt));
		}
		std::complex<double>* mssRow = rowMSS.memptr();
		std::complex<double>* qssRow = rowQSS.memptr();
		Mss->row(mssRow, is);
		Qss->row(qssRow, is);
		//delete[] qssRow;
		//delete[] mssRow;
	}

	// Calculation of MiS
	arma::mat R1(nSrcTrans, nSrcTotal);
	arma::mat x1(nSrcTrans, nSrcTotal);
	arma::mat z1(nSrcTrans, nSrcTotal);

	// total number of sources
	for (int indice = 0; indice < nSrcTrans; indice++) {
		CoordCentSourcePt_x = TransCoord_Btm(0, indice);
		CoordCentSourcePt_y = TransCoord_Btm(1, indice);
		CoordCentSourcePt_z = TransCoord_Btm(2, indice);

		// total no. of target point
		for (int i = 0; i < nSrcTotal; i++) {
			R1(indice, i) = sqrt(pow((IntfaceCoord_Cent(0, i) - CoordCentSourcePt_x), 2) + pow((IntfaceCoord_Cent(1, i) - CoordCentSourcePt_y), 2) + pow((IntfaceCoord_Cent(2, i) - CoordCentSourcePt_z), 2));
			x1(indice, i) = ((IntfaceCoord_Cent(0, i) - CoordCentSourcePt_x));
			z1(indice, i) = ((IntfaceCoord_Cent(2, i) - CoordCentSourcePt_z));

		}
	}
	complex<double> Mx1S(0, 0), Mz1S(0, 0);
	for (is = 0; is < nSrcTrans; is++) {
		arma::cx_vec colMiS(nSrcTotal);
		arma::cx_vec colQiS(nSrcTotal);

		for (jt = 0; jt < nSrcTotal; jt++) {
			Mx1S = ((x1(is, jt) * exp(img*WaveNum_P*R1(is, jt)))*(img*WaveNum_P - (1 / R1(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R1(is, jt), 2)));
			Mz1S = ((z1(is, jt) * exp(img*WaveNum_P*R1(is, jt)))*(img*WaveNum_P - (1 / R1(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R1(is, jt), 2)));

			colMiS(jt) = -Mx1S * sin(Rotation_Trans) + Mz1S * cos(Rotation_Trans);
			colQiS(jt) = (exp(img*WaveNum_P*R1(is, jt))) / (R1(is, jt));
		}
		std::complex<double>* misCol = colMiS.memptr();
		std::complex<double>* qisCol = colQiS.memptr();
		Mis->col(misCol, is);	// Mis is not required as velocity at the interface is not prescribed
		Qis->col(qisCol, is);
		//delete[] misCol;
		//delete[] qisCol;
	}

	// Calculation of MSi
	arma::mat R2(nSrcTotal, nSrcTrans);
	arma::mat x2(nSrcTotal, nSrcTrans);
	arma::mat z2(nSrcTotal, nSrcTrans);

	// total number of sources
	for (int indice = 0; indice < nSrcTotal; indice++) {
		CoordCentSourcePt_x = IntfaceCoord_Top(0, indice);
		CoordCentSourcePt_y = IntfaceCoord_Top(1, indice);
		CoordCentSourcePt_z = IntfaceCoord_Top(2, indice);

		//  total no. of target point
		for (int i = 0; i < nSrcTrans; i++) {
			R2(indice, i) = sqrt(pow(TransCoord_Cent(0, i) - CoordCentSourcePt_x, 2) + pow(TransCoord_Cent(1, i) - CoordCentSourcePt_y, 2) + pow(TransCoord_Cent(2, i) - CoordCentSourcePt_z, 2));
			x2(indice, i) = ((TransCoord_Cent(0, i) - CoordCentSourcePt_x));
			z2(indice, i) = ((TransCoord_Cent(2, i) - CoordCentSourcePt_z));
		}
	}

	complex<double> MxS1 = 0, MzS1 = 0;
	for (is = 0; is < nSrcTotal; is++) {
		arma::cx_vec colMSi(nSrcTrans);
		arma::cx_vec colQSi(nSrcTrans);

		for (jt = 0; jt < nSrcTrans; jt++) {
			MxS1 = ((x2(is, jt) * exp(img*WaveNum_P*R2(is, jt)))*(img*WaveNum_P - (1 / R2(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R2(is, jt), 2)));
			MzS1 = ((z2(is, jt) * exp(img*WaveNum_P*R2(is, jt)))*(img*WaveNum_P - (1 / R2(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R2(is, jt), 2)));

			if (InterfaceIndex == 0) {
				colMSi(jt) = -MxS1 * sin(Rotation_Trans) + MzS1 * cos(Rotation_Trans);
			}

			else if (InterfaceIndex == 1) {
				colMSi(jt) = -MxS1 * sin(Rotation_Trans) - MzS1 * cos(Rotation_Trans);
			}
			else {
				//rowMSS(jt) = 0;
				throw std::runtime_error("interface index must be 0 or 1.");
			}
			colQSi(jt) = (exp(img*WaveNum_P*R2(is, jt))) / (R2(is, jt));
		}
		std::complex<double>* msiCol = colMSi.memptr();
		std::complex<double>* qsiCol = colQSi.memptr();
		Msi->col(msiCol, is);
		Qsi->col(qsiCol, is);
		//delete[] msiCol;
		//delete[] qsiCol;
	}

	// Calculation of Mii
	arma::mat R3(nSrcTotal, nSrcTotal);
	arma::mat x3(nSrcTotal, nSrcTotal);
	arma::mat z3(nSrcTotal, nSrcTotal);

	// total number of sources
	for (int indice = 0; indice < nSrcTotal; indice++) {
		CoordCentSourcePt_x = IntfaceCoord_Top(0, indice);
		CoordCentSourcePt_y = IntfaceCoord_Top(1, indice);
		CoordCentSourcePt_z = IntfaceCoord_Top(2, indice);

		// total no. of target point
		for (int i = 0; i < nSrcTotal; i++) {
			R3(indice, i) = sqrt(pow((IntfaceCoord_Cent(0, i) - CoordCentSourcePt_x), 2) + pow((IntfaceCoord_Cent(1, i) - CoordCentSourcePt_y), 2) + pow((IntfaceCoord_Cent(2, i) - CoordCentSourcePt_z), 2));
			x3(indice, i) = ((IntfaceCoord_Cent(0, i) - CoordCentSourcePt_x));
			z3(indice, i) = ((IntfaceCoord_Cent(2, i) - CoordCentSourcePt_z));

		}
	}

	complex<double> Mx11(0, 0), Mz11(0, 0);
	for (is = 0; is < nSrcTotal; is++) {
		arma::cx_vec rowMii(nSrcTotal);
		arma::cx_vec rowQii(nSrcTotal);

		for (jt = 0; jt < nSrcTotal; jt++) {
			Mx11 = ((x3(is, jt) * exp(img*WaveNum_P*R3(is, jt)))*(img*WaveNum_P - (1 / R3(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R3(is, jt), 2)));
			Mz11 = ((z3(is, jt) * exp(img*WaveNum_P*R3(is, jt)))*(img*WaveNum_P - (1 / R3(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R3(is, jt), 2)));

			rowMii(jt) = Mx11 * sin(Rotation_Trans) + Mz11 * cos(Rotation_Trans);
			rowQii(jt) = (exp(img*WaveNum_P*R3(is, jt))) / (R3(is, jt));
		}
		std::complex<double>* miiRow = rowMii.memptr();
		std::complex<double>* qiiRow = rowQii.memptr();
		Mii->row(miiRow, is);	// Mii is not required as velocity at the interface is not prescribed
		Qii->row(qiiRow, is);
		//delete[] miiRow;
		//delete[] qiiRow;
	}
}

void PressureFluidMat::save(const std::string& path) {
	toArma(Mss).save(path + "Mss", arma::csv_ascii);
	toArma(Msi).save(path + "Msi", arma::csv_ascii);
	toArma(Qis).save(path + "Qis", arma::csv_ascii);
	toArma(Qii).save(path + "Qii", arma::csv_ascii);

	toArma(Mis.get()).save(path + "Mis", arma::csv_ascii);
	toArma(Mii.get()).save(path + "Mii", arma::csv_ascii);
	toArma(Qss.get()).save(path + "Qss", arma::csv_ascii);
	toArma(Qsi.get()).save(path + "Qsi", arma::csv_ascii);
}

vat::CxdMat& PressureFluidMat::getMSS() {
	return *Mss;
}

vat::CxdMat& PressureFluidMat::getMsi() {
	return *Msi;
}

vat::CxdMat& PressureFluidMat::getQis() {
	return *Qis;
}

vat::CxdMat& PressureFluidMat::getQii() {
	return *Qii;
}

vat::CxdMat& PressureFluidMat::getMis() {
	return *Mis;
}

vat::CxdMat& PressureFluidMat::getMii() {
	return *Mii;
}

vat::CxdMat& PressureFluidMat::getQss() {
	return *Qss;
}

vat::CxdMat& PressureFluidMat::getQsi() {
	return *Qsi;
}

LinePressureFluidMat::LinePressureFluidMat(unsigned long _nSrcTotal, unsigned long _nSrcTrans,
		unsigned long _NumLinePt, vat::DiskCache* cache, vat::MemorySize maxBlockSize) {
	nSrcTotal = _nSrcTotal;
	nSrcTrans = _nSrcTrans;
	NumLinePt = _NumLinePt;

	Mls = new arma::cx_mat(NumLinePt, nSrcTrans);
	Mli = new arma::cx_mat(NumLinePt, nSrcTotal);
	Qls = new arma::cx_mat(NumLinePt, nSrcTrans);
	Qli = new arma::cx_mat(NumLinePt, nSrcTotal);
}

LinePressureFluidMat::~LinePressureFluidMat() {
	delete Mls;
	delete Mli;
	delete Qls;
	delete Qli;
}

void LinePressureFluidMat::solve(const arma::mat& Line,
	const arma::mat& TransCoord_Source,
	const arma::mat& IntfaceCoord_Source,
	double     WaveNum_P,
	double     Fluid_rho,
	double     WaveVel_P,
	double     Rotation_Trans,
	int		   InterfaceIndex) {

	complex<double> img(0, 1);
	Rotation_Trans = Rotation_Trans * M_PI / 180;

	unsigned long is, jt;

	// Calculation of MSS
	arma::mat R(nSrcTrans, NumLinePt);
	arma::mat x(nSrcTrans, NumLinePt);
	arma::mat z(nSrcTrans, NumLinePt);
	double CoordCentSourcePt_x, CoordCentSourcePt_y, CoordCentSourcePt_z;

	for (int indice = 0; indice <nSrcTrans; indice++) {
		CoordCentSourcePt_x = TransCoord_Source(0, indice);
		CoordCentSourcePt_y = TransCoord_Source(1, indice);
		CoordCentSourcePt_z = TransCoord_Source(2, indice);

		for (int i = 0; i <NumLinePt; i++) {
			R(indice, i) = sqrt(pow((Line(0, i) - CoordCentSourcePt_x), 2) + pow((Line(1, i) - CoordCentSourcePt_y), 2) + pow((Line(2, i) - CoordCentSourcePt_z), 2));
			x(indice, i) = ((Line(0, i) - CoordCentSourcePt_x));
			z(indice, i) = ((Line(2, i) - CoordCentSourcePt_z));
		}
	}
	complex<double> MxSS(0, 0), MzSS(0, 0);
	for (is = 0; is < nSrcTrans; is++) {
		arma::cx_vec rowMSS(NumLinePt);
		arma::cx_vec rowQSS(NumLinePt);

		for (jt = 0; jt < NumLinePt; jt++) {
			MxSS = ((x(is, jt)*exp(img*WaveNum_P*R(is, jt)))*(img*WaveNum_P - (1 / R(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*pow(R(is, jt), 2));
			MzSS = ((z(is, jt) * exp(img*WaveNum_P*R(is, jt)))*(img*WaveNum_P - (1 / R(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R(is, jt), 2)));

			if (InterfaceIndex == 0) {
				rowMSS(jt) = -MxSS * sin(Rotation_Trans) + MzSS * cos(Rotation_Trans);
			}

			else if (InterfaceIndex == 1) {
				rowMSS(jt) = -MxSS * sin(Rotation_Trans) - MzSS * cos(Rotation_Trans);
			}

			else {
				throw std::runtime_error("interface index must be 0 or 1.");
			}
			rowQSS(jt) = (exp(img*WaveNum_P*R(is, jt))) / (R(is, jt));

			Mls->at(jt, is) = rowMSS(jt);
			Qls->at(jt, is) = rowQSS(jt);

			//cout<<rowQSS(jt)<<endl;
			//lineoutput->Qls->element(rowQSS(jt),jt, is);
			//lineoutput->Mls->element(rowMSS(jt),jt, is);

		}
		//rowQSS.print("rowQSS output");
		//lineoutput->Qls->col(rowQSS.memptr(), is);
		//lineoutput->Mls->col(rowMSS.memptr(), is);
		//cout<<is<<endl;
	}

	// Calculation of MSi
	arma::mat R2(nSrcTotal, NumLinePt);
	arma::mat x2(nSrcTotal, NumLinePt);
	arma::mat z2(nSrcTotal, NumLinePt);

	// total number of sources
	for (int indice = 0; indice < nSrcTotal; indice++) {
		CoordCentSourcePt_x = IntfaceCoord_Source(0, indice);
		CoordCentSourcePt_y = IntfaceCoord_Source(1, indice);
		CoordCentSourcePt_z = IntfaceCoord_Source(2, indice);

		//  total no. of target point
		for (int i = 0; i < NumLinePt; i++) {
			R2(indice, i) = sqrt(pow(Line(0, i) - CoordCentSourcePt_x, 2) + pow(Line(1, i) - CoordCentSourcePt_y, 2) + pow(Line(2, i) - CoordCentSourcePt_z, 2));
			x2(indice, i) = ((Line(0, i) - CoordCentSourcePt_x));
			z2(indice, i) = ((Line(2, i) - CoordCentSourcePt_z));


		}
	}

	complex<double> MxS1 = 0, MzS1 = 0;
	for (is = 0; is < nSrcTotal; is++) {
		arma::cx_vec colMSi(NumLinePt);
		arma::cx_vec colQSi(NumLinePt);

		for (jt = 0; jt < NumLinePt; jt++) {
			MxS1 = ((x2(is, jt) * exp(img*WaveNum_P*R2(is, jt)))*(img*WaveNum_P - (1 / R2(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R2(is, jt), 2)));
			MzS1 = ((z2(is, jt) * exp(img*WaveNum_P*R2(is, jt)))*(img*WaveNum_P - (1 / R2(is, jt)))) / (img*WaveNum_P*WaveVel_P*Fluid_rho*(pow(R2(is, jt), 2)));

			if (InterfaceIndex == 0) {
				colMSi(jt) = -MxS1 * sin(Rotation_Trans) + MzS1 * cos(Rotation_Trans);
			}

			else if (InterfaceIndex == 1) {
				colMSi(jt) = -MxS1 * sin(Rotation_Trans) - MzS1 * cos(Rotation_Trans);
			}
			else {
				throw std::runtime_error("interface index must be 0 or 1.");
			}
			colQSi(jt) = (exp(img*WaveNum_P*R2(is, jt))) / (R2(is, jt));

			Mli->at(jt, is) = colMSi(jt);
			Qli->at(jt, is) = colQSi(jt);

			//lineoutput->Qli->element(colQSi(jt),jt, is);
			//lineoutput->Mli->element(colMSi(jt),jt, is);
		}
		//lineoutput->Mli->col(colMSi.memptr(), is);
		//lineoutput->Qli->col(colQSi.memptr(), is);
	}
}

void LinePressureFluidMat::save(const std::string& path) {
	Mls->save(path + "Mls", arma::csv_ascii);
	Mli->save(path + "Mli", arma::csv_ascii);
	Qls->save(path + "Qls", arma::csv_ascii);
	Qli->save(path + "Qli", arma::csv_ascii);
}

arma::cx_mat& LinePressureFluidMat::getMls() {
	return *Mls;
}

arma::cx_mat& LinePressureFluidMat::getMli() {
	return *Mli;
}

arma::cx_mat& LinePressureFluidMat::getQls() {
	return *Qls;
}

arma::cx_mat& LinePressureFluidMat::getQli() {
	return *Qli;
}
