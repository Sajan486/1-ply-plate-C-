#include "Discretizer.h"

Discretizer::Discretizer(unsigned long numSolidFluidIntrFc) {
	IntrFcCoord_Cent    = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));

	IntrFcCoord_Top     = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));

	IntrFcCoord_Btm     = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));

	Sw_IntrFcCoord_Cent = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));

	Sw_IntrFcCoord_Top  = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));

	Sw_IntrFcCoord_Btm  = std::unique_ptr<arma::field<arma::mat>>
			(new arma::field<arma::mat>(1, numSolidFluidIntrFc));
}

Discretizer::~Discretizer() {}

void Discretizer::discretize(arma::mat&    IntrFcCoord_z,
							 double        NumSourcePt_IntrFc_x,
							 double        NumSourcePt_IntrFc_y,
							 double        Length_IntrFc_x,
							 double        Length_IntrFc_y,
							 unsigned long NumSolidFluidIntrFc,
							 double        IntrFcShift) {
	std::cout << "Start Discretizer\n";

	double NumSourceTot = NumSourcePt_IntrFc_x * NumSourcePt_IntrFc_y;			//  total number of source points (on either side of the interface)
	double IntrFcArea   = Length_IntrFc_x * Length_IntrFc_y;					//  area of the interface
	double Source_EqivR = sqrt(IntrFcArea / (NumSourceTot * 2.0 * M_PI));       //  equivalent radius of the sources
	double DistSource_x = Length_IntrFc_x / NumSourcePt_IntrFc_x;				//  distance betxeen the sources along x axis
	double DistSource_y = Length_IntrFc_y / NumSourcePt_IntrFc_y;				//  distance betxeen the sources along y axis

	int xIndex, yIndex, indice;
	
	double inivalue_x, finalvalue_x, increment_x,numterms_x, inivalue_y, finalvalue_y, increment_y, numterms_y;
	inivalue_x = -((NumSourcePt_IntrFc_x - 1) / 2) * DistSource_x + IntrFcShift;
	finalvalue_x = ((NumSourcePt_IntrFc_x - 1) / 2) * DistSource_x + IntrFcShift;
	increment_x = DistSource_x;
	numterms_x = floor((finalvalue_x - inivalue_x) / increment_x + 1);
	
	inivalue_y = -((NumSourcePt_IntrFc_y - 1) / 2) * DistSource_y + IntrFcShift;
	finalvalue_y = ((NumSourcePt_IntrFc_y - 1) / 2) * DistSource_y + IntrFcShift;
	increment_y = DistSource_y;
	numterms_y = floor((finalvalue_y - inivalue_y) / increment_y + 1);

	arma::mat X(1, numterms_x), Y(1, numterms_y), Z(IntrFcCoord_z.n_rows, 1);
	arma::mat Cent(3, NumSourcePt_IntrFc_y * NumSourcePt_IntrFc_x);
	arma::mat Top(3, NumSourcePt_IntrFc_y * NumSourcePt_IntrFc_x);
	arma::mat Btm(3, NumSourcePt_IntrFc_y * NumSourcePt_IntrFc_x);
	
	//Sweeping for Solid Interface Only
	/*
	 * Localization of sources on both faces of the fluid interface:
	 * in=interface no. index
	 */
	X(0, 0) = inivalue_x;
	Y(0, 0) = inivalue_y;
	
	for (int i = 1; i < numterms_x; i++) {
		X(0, i) = X(0, i - 1) + DistSource_x;
	}
	
	for (int i = 1; i < numterms_y; i++) {
		Y(0, i) = Y(0, i - 1) + DistSource_y;
	}
	
	for (int i = 0;i < IntrFcCoord_z.n_rows;i++) {
		Z(i, 0) = IntrFcCoord_z(i, 2);
	}

	for (int in = 0; in < NumSolidFluidIntrFc; in++) {
		for (yIndex = 0; yIndex < (int) NumSourcePt_IntrFc_y; yIndex++) {
			for (xIndex = 0; xIndex < (int) NumSourcePt_IntrFc_x; xIndex++) {

				indice = xIndex + yIndex * ((int) NumSourcePt_IntrFc_x);

				// IntrFcCoord_Cent calculation
				Cent(0, indice) = X(0, xIndex);
				Cent(1, indice) = Y(0, yIndex);
				Cent(2, indice) = Z(in, 0);
				IntrFcCoord_Cent->at(0, in) = Cent;
				
				// IntrFcCoord_Top calculation
				Top(0, indice) = X(0, xIndex);
				Top(1, indice) = Y(0, yIndex);
				Top(2, indice) = Z(in, 0) + Source_EqivR;
				IntrFcCoord_Top->at(0, in) = Top;

				// IntrFcCoord_Btm calculation
				Btm(0, indice) = X(0, xIndex);
				Btm(1, indice) = Y(0, yIndex);
				Btm(2, indice) = Z(in, 0) - Source_EqivR;
				IntrFcCoord_Btm->at(0, in) = Btm;
								
			}
		}
	}
	//IntrFcCoord_Btm->print();

	//For sweeping purpose
	double NumTrgGreenExtnd_x = (2 * NumSourcePt_IntrFc_x - 1);       // Here we take extended number of point sources for sweeping
	double NumTrgGreenExtnd_y = (2 * NumSourcePt_IntrFc_y - 1);

	inivalue_x = -((NumTrgGreenExtnd_x - 1) / 2)*DistSource_x + IntrFcShift;
	finalvalue_x = ((NumTrgGreenExtnd_x - 1) / 2)*DistSource_x + IntrFcShift;
	increment_x = DistSource_x;
	numterms_x = floor((finalvalue_x - inivalue_x) / increment_x + 1);

	inivalue_y = -((NumTrgGreenExtnd_y - 1) / 2)*DistSource_y;
	finalvalue_y = ((NumTrgGreenExtnd_y - 1) / 2)*DistSource_y;
	increment_y = DistSource_y;
	numterms_y = floor((finalvalue_y - inivalue_y) / increment_y + 1);

	arma::mat sX(1, numterms_x), sY(1, numterms_y), sZ(IntrFcCoord_z.n_rows, 1);
	arma::mat Sw_Cent(3, NumTrgGreenExtnd_y * NumTrgGreenExtnd_x);
	arma::mat Sw_Top(3, NumTrgGreenExtnd_y * NumTrgGreenExtnd_x);
	arma::mat Sw_Btm(3, NumTrgGreenExtnd_y * NumTrgGreenExtnd_x);

	sX(0, 0) = inivalue_x;
	sY(0, 0) = inivalue_y;
	
	for (int i = 1;i<numterms_x;i++) {
		sX(i) = sX(i - 1) + DistSource_x;
	}

	for (int i = 1;i<numterms_y;i++) {
		sY(i) = sY(i - 1) + DistSource_y;
	}

	for (int i = 0;i < IntrFcCoord_z.n_rows;i++) {
		sZ(i, 0) = IntrFcCoord_z(i, 2);
	}
	
	for (int in = 0; in < NumSolidFluidIntrFc; in++) {
		for (yIndex = 0; yIndex < (int)NumTrgGreenExtnd_y; yIndex++) {
			for (xIndex = 0; xIndex < (int)NumTrgGreenExtnd_x; xIndex++) {
				indice = xIndex + yIndex * ((int)NumTrgGreenExtnd_x);
				
				//IntrFcCoord_Cent calculation
				Sw_Cent(0, indice) = sX(0, xIndex);
				Sw_Cent(1, indice) = sY(0, yIndex);
				Sw_Cent(2, indice) = sZ(in, 0);
				Sw_IntrFcCoord_Cent->at(0, in) = Sw_Cent;
				
				//IntrFcCoord_Top calculation
				Sw_Top(0, indice) = sX(0, xIndex);
				Sw_Top(1, indice) = sY(0, yIndex);
				Sw_Top(2, indice) = sZ(in, 0) + Source_EqivR;
				Sw_IntrFcCoord_Top->at(0, in) = Sw_Top;
				
				//IntrFcCoord_Btm calculation
				Sw_Btm(0, indice) = sX(0, xIndex);
				Sw_Btm(1, indice) = sY(0, yIndex);
				Sw_Btm(2, indice) = sZ(in, 0) - Source_EqivR;
				Sw_IntrFcCoord_Btm->at(in) = Sw_Btm;
			}
		}
	}
	std::cout << "End Discretizer\n";
}

const arma::field<arma::mat>& Discretizer::getIntrFcCoord_Cent() {
	return *IntrFcCoord_Cent;
}

const arma::field<arma::mat>& Discretizer::getIntrFcCoord_Top() {
	return *IntrFcCoord_Top;
}

const arma::field<arma::mat>& Discretizer::getIntrFcCoord_Btm() {
	return *IntrFcCoord_Btm;
}

const arma::field<arma::mat>& Discretizer::getSw_IntrFcCoord_Cent() {
	return *Sw_IntrFcCoord_Cent;
}

const arma::field<arma::mat>& Discretizer::getSw_IntrFcCoord_Top() {
	return *Sw_IntrFcCoord_Top;
}

const arma::field<arma::mat>& Discretizer::getSw_IntrFcCoord_Btm() {
	return *Sw_IntrFcCoord_Btm;
}