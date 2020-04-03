#ifndef GEOMSENSCIRCLE_H_
#define GEOMSENSCIRCLE_H_

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <armadillo>
#include "Config.h"

using namespace arma;
using namespace std;

class DiscretizeTransducerOutput {
public:
	arma::mat TransCoord_Cent0,
			  TransCoord_Top0,
			  TransCoord_Btm0,
			  TransCoord_Cent1,
			  TransCoord_Top1,
			  TransCoord_Btm1;

	double NumSourcePt_Trans;
};

DiscretizeTransducerOutput DiscretizeTransducer(Transducer& trans, Geom& geom);
#endif
