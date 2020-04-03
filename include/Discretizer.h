#ifndef DISCRETIZER_H_
#define DISCRETIZER_H_

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <armadillo>
#include "Config.h"
#include <memory>

class Discretizer {
public:
	Discretizer(unsigned long numSolidFluidIntrFc);

	Discretizer(Discretizer const &) = delete;

	Discretizer &operator=(Discretizer const &) = delete;

	Discretizer(Discretizer &&obj) :
			IntrFcCoord_Cent(std::move(obj.IntrFcCoord_Cent)),
			IntrFcCoord_Btm (std::move(obj.IntrFcCoord_Btm)),
			IntrFcCoord_Top (std::move(obj.IntrFcCoord_Top)),
			Sw_IntrFcCoord_Cent(std::move(obj.Sw_IntrFcCoord_Cent)),
			Sw_IntrFcCoord_Btm (std::move(obj.Sw_IntrFcCoord_Btm)),
			Sw_IntrFcCoord_Top (std::move(obj.Sw_IntrFcCoord_Top)) {}

	Discretizer &operator=(Discretizer &&obj) {
		if (this != &obj) {
			IntrFcCoord_Btm 	= std::move(obj.IntrFcCoord_Btm);
			IntrFcCoord_Top 	= std::move(obj.IntrFcCoord_Top);
			IntrFcCoord_Cent 	= std::move(obj.IntrFcCoord_Cent);
			Sw_IntrFcCoord_Btm	= std::move(obj.Sw_IntrFcCoord_Btm);
			Sw_IntrFcCoord_Top	= std::move(obj.Sw_IntrFcCoord_Top);
			Sw_IntrFcCoord_Cent = std::move(obj.Sw_IntrFcCoord_Cent);
		}
		return *this;
	}
	~Discretizer();

	void save(const std::string& path = "");

	void discretize(arma::mat&    IntrFcCoord_z,
				    double        NumSourcePt_IntrFc_x,
					double        NumSourcePt_IntrFc_y,
				    double        Length_IntrFc_x,
					double        Length_IntrFc_y,
					unsigned long NumSolidFluidIntrFc,
					double        IntrFcShift);

	const arma::field<arma::mat>& getIntrFcCoord_Cent();
	const arma::field<arma::mat>& getIntrFcCoord_Top();
	const arma::field<arma::mat>& getIntrFcCoord_Btm();
	const arma::field<arma::mat>& getSw_IntrFcCoord_Cent();
	const arma::field<arma::mat>& getSw_IntrFcCoord_Top();
	const arma::field<arma::mat>& getSw_IntrFcCoord_Btm();

private:
	std::unique_ptr<arma::field<arma::mat>> IntrFcCoord_Cent;
	std::unique_ptr<arma::field<arma::mat>> IntrFcCoord_Top;
	std::unique_ptr<arma::field<arma::mat>> IntrFcCoord_Btm;
	std::unique_ptr<arma::field<arma::mat>> Sw_IntrFcCoord_Cent;
	std::unique_ptr<arma::field<arma::mat>> Sw_IntrFcCoord_Top;
	std::unique_ptr<arma::field<arma::mat>> Sw_IntrFcCoord_Btm;
};

#endif
