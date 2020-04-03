#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <armadillo>
#include "Matrix.h"
#include <string>
#include <stdlib.h>
#include <fstream>
#include <memory>


class Geom {
public:
	Geom();
	~Geom();

	/*
	 * + No.of solid layers in multilayer solids
	 * + No.of Fluid layers on boths side of solid faces
	 * + No.of interfaces
	 * + No.of fluid solid interface
	 */
	double NumSolidLay;
	double NumFluidLay;
	double NumIntrFc;
	double NumSolidFluidIntrFc;

	/*
	 * + Number source points(on either side) on the surface along the x
	 * axis preferably odd number
	 * + Number source points(on either side) on the surface along the y
	 * axis preferably odd number
	 * + Number of source points total
	 * + Angle for Christofel solution
	 */
	double NumSourcePt_IntrFc_x;
	double NumSourcePt_IntrFc_y;
	double NumSourcePt_IntrFc_Total;
	double AngTestPt;

	/*
	 * + Length of the interface along the x axis in mm
	 * + Length of the interface along the y axis in mm
	 */
	double Length_IntrFc_x;
	double Length_IntrFc_y;

	/*
	 * + Distance between transducer and Interface 1 (fluid - solid
	 * interface)
	 * + Distance between transducer and Interface 2 (solid - solid interface)
	 * + Shift of Receiver transducer along X axis
	 * + Distance between transducer and Interfaces
	 */
	double Dist_IntrFc1;
	double Dist_IntrFc2;
	double Dist_2ndTrans;
	double ReceiverShift;

	/*
	 * + Distance between two point sources along x - axis
	 */
	double DistSource_x;
	double DistSource_y;

	arma::mat& getDist_IntrFc();
	arma::mat& getIntrFcCoord_z();

	void init(double IntrFcShift);

private:
	std::unique_ptr<arma::mat> Dist_IntrFc;
	std::unique_ptr<arma::mat> IntrFcCoord_z;
};

class Transducer {
public:
	Transducer();
	~Transducer();

	// Number of transducers
	double NumTrans;

	/*
	 * + Inner radius of transducer
	 * + Outer radius of transducer
	 * + Number of transducer source point
	 */
	double InnerR_Trans;
	double OuterR_Trans;
	double NumSourcePt_Trans;

	// Percentage of shift of interface coordinate towards right.
	double IntrFcShift;

	/*
	 * Rotation of the trasnsducers
	 * + Angle of rotattion of transducer 1, i.e inclination of the
	 * transducer anticlockwise
	 * + Angle of rotattion of transducer 2, i.e inclination of the
	 * transducer clockwise
	 */
	double Rotation_Trans1;
	double Rotation_Trans2;

	/*
	 * + Actuation frequency in MHz
	 * + Actuation ffrequency in rads / sec
	 * + Velocity of transducer(Source or Transmitter) face enter magnitude
	 * only
	 * + Velocity of transducer(Source or Receiver) face enter magnitude
	 *  only
	 */
	double freq;
	double w;
	double Vso;
	double Vto;

	arma::mat& getTransCoord_z();

	void init(double Length_IntrFc_x, double Dist_2ndTrans);

private:
	// Origin of the transducer
	std::unique_ptr<arma::mat> TransCoord_z;
};

class Wavefield {
public:
	Wavefield();
	/*
	* Number of coordinates taken for plotting
	*/
	double NumTarget_z;

	/*
	 * Planar coordinates for plot modes 1, 2, 3
	 */
	double X_PlaneCoord;
	double Y_PlaneCoord;
	double Z_PlaneCoord;

	void init(double Dist_IntrFc1);
};

class Fluid {
public:
	/*
	* + Density of fluid layers
	* + P - wave speed
	* + P - Wave number
	*/

	Fluid();
	~Fluid();

	double Fluid_rho;

	double WaveVel_P;
	double WaveNum_P;

	arma::mat& getFluidPoint();

	void init(double freq);

private:
	std::unique_ptr<arma::mat> FluidPoint;
};

class Solid {
public:
	Solid();
	~Solid();

	// Density of solid layers
	double Solid_rho;

	arma::mat& getC();
private:
	// Constitutive stiffness matrix
	std::unique_ptr<arma::mat> C;
};

class Timedomain {
public:
	Timedomain();

	double SampRate;
	double delTime;
	double NumSampPt;
	double CentFreq;
	double H;
	double k;
	double NumCycles;
};

class Config {
public:
	Config();
	~Config();

	bool validationMode;
	std::string validationFolder;

	bool doublePrecision;

	// GPU params
	bool enableGPU;
	std::vector<int> targetDevices;

	// cache params
	std::string cachePath;
	vat::MemorySize cacheSize;
	vat::MemorySize maxIO;
	vat::MemorySize maxBlockSize;

	std::vector<int>* plotModes;
	std::string plotOutputFolder;

	Transducer&	getTransducer();
	Geom&		getGeom();
	Wavefield&	getWavefield();
	Fluid&		getFluid();
	Solid&		getSolid();
	Timedomain&	getTimedomain();

private:
	std::unique_ptr<Transducer> transducer;
	std::unique_ptr<Geom> 		geom;
	std::unique_ptr<Wavefield>	wavefield;
	std::unique_ptr<Fluid> 		fluid;
	std::unique_ptr<Solid> 		solid;
	std::unique_ptr<Timedomain> timedomain;
};
#endif
