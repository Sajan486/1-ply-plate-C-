#include "Config.h"

Geom::Geom()  {
	NumSolidLay = 1;
	NumFluidLay = 2;
	NumIntrFc = NumSolidLay + 1;
	NumSolidFluidIntrFc = 2;

	NumSourcePt_IntrFc_x = 115;
	NumSourcePt_IntrFc_y = 1;
	NumSourcePt_IntrFc_Total = NumSourcePt_IntrFc_x * NumSourcePt_IntrFc_y;
	AngTestPt = 1;

	Length_IntrFc_x = 40;
	Length_IntrFc_y = (Length_IntrFc_x * NumSourcePt_IntrFc_y) / NumSourcePt_IntrFc_x;

	Dist_IntrFc1 = 5;
	Dist_IntrFc2 = 8;
	Dist_2ndTrans = Dist_IntrFc2 + Dist_IntrFc1;
	ReceiverShift= 0;

	DistSource_x = Length_IntrFc_x / NumSourcePt_IntrFc_x;
	DistSource_y = Length_IntrFc_y / NumSourcePt_IntrFc_y;

	Dist_IntrFc = std::unique_ptr<arma::mat> (new arma::mat(1, 4));
	IntrFcCoord_z = std::unique_ptr<arma::mat> (new arma::mat(2, 3));

	(*Dist_IntrFc) << 0 << Dist_IntrFc1 << Dist_IntrFc2 << Dist_2ndTrans << arma::endr;

	(*IntrFcCoord_z) << 0 << 0 << Dist_IntrFc1 << arma::endr
				     << 0 << 0 << Dist_IntrFc2 << arma::endr;
}

Geom::~Geom() {}

void Geom::init(double IntrFcShift) {
	ReceiverShift= -2 * (IntrFcShift / 100) * Length_IntrFc_x;
}

arma::mat& Geom::getDist_IntrFc() {
	return *Dist_IntrFc;
}

arma::mat& Geom::getIntrFcCoord_z() {
	return *IntrFcCoord_z;
}

Transducer::Transducer() {
	NumTrans = 2;

	InnerR_Trans = 0;
	OuterR_Trans = 2;
	NumSourcePt_Trans = 100;

	IntrFcShift = 0;

	Rotation_Trans1 = 0;
	Rotation_Trans2 = 0;

	freq = 0.01;
	Vso = 1;
	Vto = 1;

	w = 2 * M_PI * freq;

	TransCoord_z = std::unique_ptr<arma::mat> (new arma::mat(2, 3));
}

Transducer::~Transducer() {}

void Transducer::init(double Length_IntrFc_x, double Dist_2ndTrans) {
	IntrFcShift = -(IntrFcShift / 100) * Length_IntrFc_x;
	(*TransCoord_z) << 0 << 0 << 0	<< arma::endr
					<< 0 << 0 << Dist_2ndTrans << arma::endr;
}

arma::mat& Transducer::getTransCoord_z() {
	return *TransCoord_z;
}

Fluid::Fluid() {
	Fluid_rho = 1;
	WaveVel_P = 1.48;

	WaveNum_P = 0.0;

	FluidPoint = std::unique_ptr<arma::mat> (new arma::mat(3, 1));
	(*FluidPoint) << 0  << arma::endr
			   	  << 0  << arma::endr
			   	  << 13 << arma::endr;
}

Fluid::~Fluid() {}

void Fluid::init(double freq) {
	WaveNum_P = (2 * M_PI * freq) / WaveVel_P;
}

arma::mat& Fluid::getFluidPoint() {
	return *FluidPoint;
}

Solid::Solid() {
	Solid_rho = 1.56; // transversely isotropic/monoclinic
	//Solid_rho = 1.5; // orthotropic
	C = std::unique_ptr<arma::mat> (new arma::mat(6, 6));

	// Transversely Isotropic
	/*
	(*C) << 143.8 << 6.2  << 6.2  << 0   << 0   << 0   << arma::endr
	  << 6.2   << 13.3 << 6.5  << 0   << 0   << 0      << arma::endr
      << 6.2   << 6.5  << 13.3 << 0   << 0   << 0  	   << arma::endr
      << 0     << 0    << 0    << 3.4 << 0   << 0      << arma::endr
      << 0     << 0    << 0    << 0   << 5.7 << 0      << arma::endr
      << 0     << 0    << 0    << 0   << 0   << 5.7    << arma::endr;
	//*/  
	
	// Fully Orthtropic
	/*/
	(*C) << 70 << 23.9 << 6.2  << 0   << 0   << 0   << arma::endr
	  << 23.9  << 33   << 6.8  << 0   << 0   << 0      << arma::endr
      << 6.2   << 6.8  << 14.7 << 0   << 0   << 0  	   << arma::endr
      << 0     << 0    << 0    << 4.2 << 0   << 0      << arma::endr
      << 0     << 0    << 0    << 0   << 4.7 << 0      << arma::endr
      << 0     << 0    << 0    << 0   << 0   << 21.9    << arma::endr;
	//*/
	// Monoclinic
	//*/
	(*C) << 102.6 << 24.1  << 6.3  << 0   << 0   << 40   << arma::endr
	  << 24.1   << 18.7 << 6.4   << 0    << 0    << 10      << arma::endr
      << 6.3    << 6.4  << 13.3  << 0    << 0    << -0.1    << arma::endr
      << 0      << 0    << 0     << 3.8  << 0.9  << 0       << arma::endr
      << 0      << 0    << 0     << 0.9  << 5.3  << 0       << arma::endr
      << 40     << 10   << -0.1  << 0    << 0    << 23.6    << arma::endr;
	//*/
	  
	  
}

Solid::~Solid() {}

arma::mat& Solid::getC() {
	return *C;
}

Timedomain::Timedomain() {
	SampRate = 5;        // Sampling Rate(MS / sec)
	NumSampPt = 256; //256;   // No.of sample point
	delTime = 1 / (SampRate);

	/*
	// Pre allocation of memory
	double TimeStamp = zeros(NumSampPt, 1);
	double ForceTimeSignal = zeros(NumSampPt, 1);
	double signal_tb = zeros(NumSampPt, 1);         // Signal with a sngle frequency content
	double Signal = zeros(NumSampPt, 2);
	*/
	CentFreq = 1; // Central Frequency in MHz

	// Input Signal data
	H = 1e-3; // Unit is in m.
	// Signal Parameters
	k = 5000; // A signal shape factor
	NumCycles = 5; // Number of cycles to be generated
}

Wavefield::Wavefield() {
	NumTarget_z = 31;
	X_PlaneCoord = 0.0;
	Y_PlaneCoord = 0.0;
	Z_PlaneCoord = 0.0;
}

void Wavefield::init(double Dist_IntrFc1) {
	Z_PlaneCoord = Dist_IntrFc1 + 0.1;
}


Config::Config() :
		cacheSize(15.0, vat::GB),
		maxIO(1.0, vat::GB),
		maxBlockSize(0.1, vat::GB)
{
	validationMode 	 = false;
	validationFolder = "/home/ubuntu/DPSM/validation/outputs";

	doublePrecision = false;
	enableGPU 		= true;
	targetDevices = {-1};

	cachePath = "/cache";

	plotModes = new std::vector<int>;
	//plotModes->push_back(1);
	//plotModes->push_back(2);
	plotModes->push_back(3); 

	plotOutputFolder = "/home/ubuntu/Results";


	transducer	= std::unique_ptr<Transducer> (new Transducer());
	geom		= std::unique_ptr<Geom> (new Geom());
	wavefield	= std::unique_ptr<Wavefield> (new Wavefield());
	fluid		= std::unique_ptr<Fluid> (new Fluid());
	solid		= std::unique_ptr<Solid> (new Solid());
	timedomain	= std::unique_ptr<Timedomain> (new Timedomain());

	geom->init(transducer->IntrFcShift);
	transducer->init(geom->Length_IntrFc_x, geom->Dist_2ndTrans);
	wavefield->init(geom->Dist_IntrFc1);
	fluid->init(transducer->freq);
}

Config::~Config() {}

Geom& Config::getGeom() {
	return *geom;
}

Transducer& Config::getTransducer() {
	return *transducer;
}

Wavefield& Config::getWavefield() {
	return *wavefield;
}

Timedomain& Config::getTimedomain() {
	return *timedomain;
}

Solid& Config::getSolid() {
	return *solid;
}

Fluid& Config::getFluid() {
	return *fluid;
}
