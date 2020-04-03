#include <stdio.h>
#include <math.h>
#include "GeomSensCircl.h"

DiscretizeTransducerOutput DiscretizeTransducer(Transducer& trans, Geom& geom) {
	double NumTrans          = trans.NumTrans;
	double InnerR_Trans      = trans.InnerR_Trans;
	double OuterR_Trans      = trans.OuterR_Trans;
	double NumSourcePt_Trans = trans.NumSourcePt_Trans;

	double Rotation_Trans1   = trans.Rotation_Trans1;
	double Rotation_Trans2   = trans.Rotation_Trans2;
	arma::mat TransCoord_z   = trans.getTransCoord_z();

	double Dist_IntrFc2      = geom.Dist_IntrFc2;

	double S = M_PI * (OuterR_Trans * OuterR_Trans - InnerR_Trans * InnerR_Trans);	//  Total surface area of the transducer
	double ds = S / NumSourcePt_Trans;												//  Surface area of each source
	double Source_EqivR = sqrt(S / (NumSourcePt_Trans * 2.0 * M_PI));				//  Equivalent radius of the sources

	if (InnerR_Trans == 0.0) {
		InnerR_Trans = Source_EqivR * sqrt(2);
	}
		
	double Mopt = (OuterR_Trans - InnerR_Trans) / (Source_EqivR * sqrt(2 * M_PI));
	int M = round(Mopt);															//  Number of annular ring
	double deltaR = (OuterR_Trans - InnerR_Trans) / M;								//  Thickness of each annular ring
	mat R(1, M), Dteta(1, M), W(1, M);
	NumSourcePt_Trans = 0;
	
	for (int i = 0; i < (int) M; i++) {
		R(0,i) = InnerR_Trans + ((double)i + 1 - 0.5) * deltaR;						//  Radius of each annular ring
		Dteta(0, i) = ds / (deltaR * R(0, i));
		W(0, i) = 2 * M_PI / Dteta(0, i);											//  Number of sources in each annular ring
		NumSourcePt_Trans = NumSourcePt_Trans + round(W(0, i));    					//	Total number of sources
		W(0, i) = round(W(0, i));
		Dteta(0, i) = 2.0 * M_PI / W(0, i);	  //  new ds/R*deltaR
	}
	
	mat rcoord(1, NumSourcePt_Trans), imcoord(1, NumSourcePt_Trans);
	
	int indice = 0;
	
for (int an = 0; an < M; an++) {    //  an  index for number of annular ring
	for (int s = 0; s < (int)W(0, an); s++) {      //  s index for number of sources
			
		rcoord(0, indice) = R(0, an) * cos(((double)s+1 - 0.5) * Dteta(0, an));
		imcoord(0, indice) = R(0, an) * (sin(((double)s+1 - 0.5) * Dteta(0, an)));
		indice = indice + 1;
	}
}

//==========================================================================================================================
			//  coordinates of the points(center of the sources) located on and either side of the interface  \\
//==========================================================================================================================
field<mat> coord(1, NumTrans);
field<mat> coord1(1, NumTrans);
field<mat> coord2(1, NumTrans);

mat TransCoord_Cent0(3, NumSourcePt_Trans);
mat TransCoord_Top0(3, NumSourcePt_Trans);
mat TransCoord_Btm0(3, NumSourcePt_Trans);

mat TransCoord_Cent1(3, NumSourcePt_Trans);
mat TransCoord_Top1(3, NumSourcePt_Trans);
mat TransCoord_Btm1(3, NumSourcePt_Trans);

mat coordvara(3, NumSourcePt_Trans), coordvarb(3, NumSourcePt_Trans), coordvarc(3, NumSourcePt_Trans);
mat coordvar1(3, NumSourcePt_Trans);

for (int in = 0; in < (int)NumTrans; in++) {
	for (int ind = 0; ind < (int)NumSourcePt_Trans; ind++) {
		coordvara(0, ind) = rcoord(0, ind);
		coordvara(1, ind) = imcoord(0, ind);
		coordvara(2, ind) = trans.getTransCoord_z()(in, 2);
		coord(0, in) = coordvara;
		
		coordvarb(0, ind) = rcoord(0, ind);
		coordvarb(1, ind) = imcoord(0, ind);
		coordvarb(2, ind) = trans.getTransCoord_z()(in, 2)+ Source_EqivR;
		coord1(0, in) = coordvarb;

		coordvarc(0, ind) = rcoord(0, ind);
		coordvarc(1, ind) = imcoord(0, ind);
		coordvarc(2, ind) = trans.getTransCoord_z()(in, 2) - Source_EqivR;
		coord2(0, in) = coordvarc;
	}
}

Rotation_Trans1 = Rotation_Trans1*M_PI / 180;
Rotation_Trans2 = Rotation_Trans2*M_PI / 180;
double c1 = cos(Rotation_Trans1);
double c2 = cos(Rotation_Trans2);
double s1 = sin(Rotation_Trans1);
double s2 = sin(Rotation_Trans2);
//cout << "coord(0, 0)(0, 0)		" << coord(0, 0)(0, 0) << "	" << "" << coord(0, 0)(2, 0) << "\n	" << endl;
//===================================================================================================================================
//		defining triplet source coordinate		
//===================================================================================================================================
// NOT REQUIRED FOR FLUID REFER TO SOLID, HOW TO DEFINE TRIPLET.

//===================================================================================================================================
// Defining coordinate due to rotation
//===================================================================================================================================
for (int indice = 0; indice < (int)NumSourcePt_Trans; indice++) {

	TransCoord_Cent0(0, indice) = c1*coord(0, 0)(0,indice) - s1*coord(0,0)(2,indice);
	TransCoord_Cent0(1, indice) = coord(0, 0)(1, indice);
	TransCoord_Cent0(2, indice) = s1*coord(0, 0)(0, indice) + c1*coord(0, 0)(2, indice);

	TransCoord_Top0(0, indice) = c1*coord1(0, 0)(0, indice) - s1*coord1(0, 0)(2, indice);
	TransCoord_Top0(1, indice) = coord1(0, 0)(1, indice);
	TransCoord_Top0(2, indice) = s1*coord1(0, 0)(0, indice) + c1*coord1(0, 0)(2, indice);
		
	TransCoord_Btm0(0, indice) = c1*coord2(0, 0)(0, indice) - s1*coord2(0, 0)(2, indice);
	TransCoord_Btm0(1, indice) = coord2(0, 0)(1, indice);
	TransCoord_Btm0(2, indice) = s1*coord2(0, 0)(0, indice) + c1*coord2(0, 0)(2, indice);
	
	TransCoord_Cent1(0, indice) = c2 * coord(0, 1)(0, indice) - s2 * coord(0, 1)(2, indice) + s2 * Dist_IntrFc2;
	TransCoord_Cent1(1, indice) = coord(0, 1)(1, indice);
	TransCoord_Cent1(2, indice) = s2 * coord(0, 1)(0, indice) + c2 * coord(0, 1)(2, indice) + (1 - c2) * Dist_IntrFc2;

	TransCoord_Top1(0, indice) = c2 * coord1(0, 1)(0, indice) - s2 * coord1(0, 1)(2, indice) + s2 * Dist_IntrFc2;
	TransCoord_Top1(1, indice) = coord1(0, 1)(1, indice);
	TransCoord_Top1(2, indice) = s2 * coord1(0, 1)(0, indice) + c2 * coord1(0, 1)(2, indice) + (1 - c2) * Dist_IntrFc2;

	TransCoord_Btm1(0, indice) = c2 * coord2(0, 1)(0, indice) - s2 * coord2(0, 1)(2, indice) + s2 * Dist_IntrFc2;
	TransCoord_Btm1(1, indice) = coord2(0, 1)(1, indice);
	TransCoord_Btm1(2, indice) = s2 * coord2(0, 1)(0, indice) + c2 * coord2(0, 1)(2, indice) + (1 - c2) * Dist_IntrFc2;

}
DiscretizeTransducerOutput value;

value.TransCoord_Cent0 = TransCoord_Cent0;
value.TransCoord_Top0 = TransCoord_Top0;
value.TransCoord_Btm0 = TransCoord_Btm0;
value.TransCoord_Cent1 = TransCoord_Cent1;
value.TransCoord_Top1 = TransCoord_Top1;
value.TransCoord_Btm1 = TransCoord_Btm1;
value.NumSourcePt_Trans = NumSourcePt_Trans;
return value;

}

