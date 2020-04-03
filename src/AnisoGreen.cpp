#include "AnisoGreen.h"

template<typename F>
AnisoGreen<F>::AnisoGreen(unsigned long nSrcTot, vat::DiskCache* cache, vat::MemorySize blockSize) {
	cData = new vat::CxMat<F>**[9];
	for (int i = 0; i < 9; i++) {
		// Allocate space for disk formatted slices
		cData[i] = new vat::CxMat<F>*[3];
		for (int j = 0; j < 3; j++) {
			cData[i][j] = new vat::CxMat<F>(cache, nSrcTot, nSrcTot, blockSize);
		}
	}
	cDS1 = cData[0]; cDS2 = cData[1]; cDS3 = cData[2];
	cS11 = cData[3]; cS22 = cData[4]; cS33 = cData[5];
	cS32 = cData[6]; cS31 = cData[7]; cS12 = cData[8];
}

template<typename F>
AnisoGreen<F>::~AnisoGreen() {
	delete[] cDS3;
	delete[] cS31;
	delete[] cS32;
	delete[] cS33;
	delete[] cData;
	/*for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			if (cData[i][j] != nullptr) delete cData[i][j];
		}
		if (cData[i] != nullptr) delete[] cData[i];
	}
	delete[] cData;
	*/
}


template<typename F>
void AnisoGreen<F>::solve(const arma::mat*          Sw_IntrFcCoord_Cent,
			              const arma::mat*          CoordCentSourcePt,
		                  ChristofelSphere& 		cSphere,
		                  const arma::mat*          C,
		                  F                         nSrcX,
		                  F                         nSrcY,
		                  F                         Solid_rho,
		                  F                         freq,
		                  F                         dTheta,
		                  F                         dPhi,
		                  Config&					config,
		                  vat::DiskCache*			cache) {

	double tolerance = 1e-8;

	/*
	 * Note that though SolidGreen is templated, the precision is forced by
	 * configuration settings, until a later version
	 */

	// Initialization

	long nSrcTot = (long)nSrcX * (long)nSrcY;
	unsigned long nTrgX = (2 * nSrcX - 1);
	unsigned long nTrgY = (2 * nSrcY - 1);

	long totalTarget = nTrgX * nTrgY;
	arma::cx_mat TR(1lu, 3lu * totalTarget);

	long sweepIdx;
	for (int yIdx = 0; yIdx < nTrgY; yIdx++) {
		for (int xIdx = 0; xIdx < nTrgX; xIdx++) {
			sweepIdx =  xIdx + yIdx * nTrgX;

			for (int col = 0; col < 3; col++) {
				std::complex<double> val = Sw_IntrFcCoord_Cent->at(col, sweepIdx) - CoordCentSourcePt->at(0, col);
				bool t = abs(val) < tolerance; // Careful with this.
				if (t) val.real(0);
				TR.col(3 * sweepIdx + col) = val;
			}
		}
	}

	//TR.print();

	SolidGreen<double>* sg = new SolidGreen<double>(nTrgX, nTrgY, cache);
	sg->solve(TR, *C, cSphere, Solid_rho, freq, dTheta, dPhi, config, -1);

	for (unsigned long trgCol = 0lu; trgCol < nSrcY; trgCol++) {
		for (unsigned long trgRow = 0lu; trgRow < nSrcX; trgRow++) {

			unsigned long globalIdx = trgRow + trgCol * nSrcX;
			unsigned long endTrgRow = trgRow + nSrcX - 1lu;
			unsigned long endTrgCol = trgCol + nSrcY - 1lu;
			unsigned long col = nSrcTot - globalIdx - 1lu;

			for (unsigned long i = 0lu; i < 9lu; i++) {

				arma::cx_cube subCube = sg->data[i]->tube(trgRow, trgCol, endTrgRow, endTrgCol);
				subCube.reshape(nSrcX * nSrcY, 1lu, 3lu);

				for (unsigned long slice = 0lu; slice < 3lu; slice++) {
					std::complex<F>* memSlice = (std::complex<F>*)subCube.slice(slice).memptr();
					cData[i][slice]->col(memSlice, col);
				//	delete[] memSlice;
				}
			}
		}
	}
	delete sg;
}