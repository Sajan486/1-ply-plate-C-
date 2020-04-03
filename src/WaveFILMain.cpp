#include "Discretizer.h"
#include "GeomSensCircl.h"
#include "PressureFluidMat.h"
#include "DisFluidMat.h"
#include "Settings.h"
#include "AnisoGreen.h"
#include "Solve.h"
#include <memory>
#include <chrono>
#include <omp.h>
#include <iterator>

typedef unsigned long ul;

int main(int argc, char** argv) {
	std::cout << "Launching time domain code.\n";
	
	std::chrono::time_point<std::chrono::system_clock> starttime, endtime; 
  
    starttime = std::chrono::system_clock::now();
	
	// Define calculation precision for key components
	typedef double F;

	vat::DiskCache* cache;
	Config config;

	// Apply settings file if provided
	if (argc > 1) {
		std::cout << "Settings file found\n";
		Settings::apply(argv[1], config);
	}

	try {
		std::cout << "Allocating space for disk cache...\n";
		cache = new vat::DiskCache(config.cachePath, config.cacheSize, config.maxIO, true);
		std::cout << "Disk space allocated.\n";
	}
	catch (char* msg) {
		printf("%s\n", msg);
	}

	/*
	 * Frequency calculations are placed on N separate threads.
	 * Note: DiskSpaceCost = numThreads * DiskSpacePerFrequency.
	 */
	int numThreads = 1;

	std::cout << "Initializing frequency parameters.\n";
	arma::mat TimeStamp 	  	(config.getTimedomain().NumSampPt, 1);
	arma::mat ForceTimeSignal	(config.getTimedomain().NumSampPt, 1, arma::fill::zeros);
	arma::mat signal_tb			(config.getTimedomain().NumSampPt, 1);

	const double OmegaC = 2 * M_PI * config.getTimedomain().CentFreq; // Central Frequency in MHz
	const double Tao	= config.getTimedomain().NumCycles / config.getTimedomain().CentFreq;      // Total band of pulse

	const double Tao0 	= (Tao + 5) / 2;
	const double p1		= pow((config.getTimedomain().k * config.getTimedomain().H *
			config.getTimedomain().CentFreq / config.getTimedomain().NumCycles), 2); // A factor

	for (int i = 0; i < config.getTimedomain().NumSampPt; i++) {
		TimeStamp(i) = (i) * config.getTimedomain().delTime;
		signal_tb(i) = exp(-p1 * pow((TimeStamp(i) - Tao0), 2) / 2) * sin(OmegaC*TimeStamp(i));
	}
	ForceTimeSignal = ForceTimeSignal + signal_tb;

	// Frequency Transform of the Signal
	arma::cx_mat FwSignal = fft(ForceTimeSignal); // Frequency Amplitude Signal in frequency domain
	arma::mat delW(config.getTimedomain().NumSampPt, 1); // dw for frequency domain
	for (int i = 0; i < config.getTimedomain().NumSampPt; i++) {
		delW(i) = i / (config.getTimedomain().NumSampPt * config.getTimedomain().delTime);
	}

	arma::mat f = delW.rows(0, config.getTimedomain().NumSampPt / 2 - 1); // Store half of the generated 'Actuation frequency' vector due to symmetricity about Nyquist Freq
	arma::cx_mat frAmp = FwSignal.rows(0, config.getTimedomain().NumSampPt / 2 - 1); // Store half of the generated 'Amplitude' vector due to symmetricity about Nyquist Freq
	//f.save(getTimePath(timePoint, path, "f"), csv_ascii);
	//frAmp.save(getTimePath(timePoint, path, "frAmp"), csv_ascii);
	
	std::cout << "Starting openmp threads.\n";

		#pragma omp parallel num_threads(numThreads)
	{
		#pragma omp for
		for (ul i = 0; i < numThreads; i++) {
			int frequencies = f.n_rows / numThreads;

			if (i == numThreads - 1) {
				frequencies = f.n_rows - (frequencies * i);
			}

			std::cout << "Getting point source coordinates for both interfaces and transducers.\n";
			DiscretizeTransducerOutput gs = DiscretizeTransducer(config.getTransducer(), config.getGeom());

			const unsigned long nTr = gs.NumSourcePt_Trans;
			const unsigned long nTo = config.getGeom().NumSourcePt_IntrFc_Total;
			const unsigned long gmWidth = 2 * nTr + 8lu * nTo;

			/*
			 * Set matrix views of the Green's Matrix, or references to different
			 * chunks of the matrix. These references are then passed into their
			 * respective functions to be populated appropriately
			 */
			const unsigned long b1 = nTr;
			const unsigned long b2 = nTr + 1lu * nTo;
			const unsigned long b3 = nTr + 2lu * nTo;
			const unsigned long b4 = nTr + 3lu * nTo;
			const unsigned long b5 = nTr + 4lu * nTo;
			const unsigned long b6 = nTr + 5lu * nTo;
			const unsigned long b7 = nTr + 6lu * nTo;
			const unsigned long b8 = nTr + 7lu * nTo;
			const unsigned long b9 = nTr + 8lu * nTo;
			const unsigned long b10 = 2 * nTr + 8lu * nTo;


			std::unique_ptr<Discretizer> dscrt (new Discretizer(config.getGeom().NumSolidFluidIntrFc));
			dscrt->discretize(config.getGeom().getIntrFcCoord_z(),
							config.getGeom().NumSourcePt_IntrFc_x,
							config.getGeom().NumSourcePt_IntrFc_y,
							config.getGeom().Length_IntrFc_x,
							config.getGeom().Length_IntrFc_y,
							config.getGeom().NumSolidFluidIntrFc,
							config.getTransducer().IntrFcShift);

			const arma::mat IntrFcCoord_Cent0		= dscrt->getIntrFcCoord_Cent().at(0, 0);
			const arma::mat IntrFcCoord_Top0		= dscrt->getIntrFcCoord_Top().at(0, 0);
			const arma::mat IntrFcCoord_Btm0		= dscrt->getIntrFcCoord_Btm().at(0, 0);

			const arma::mat Sw_IntrFcCoord_Cent0	= dscrt->getSw_IntrFcCoord_Cent().at(0, 0);
			const arma::mat Sw_IntrFcCoord_Top0		= dscrt->getSw_IntrFcCoord_Top().at(0, 0);
			const arma::mat Sw_IntrFcCoord_Btm0		= dscrt->getSw_IntrFcCoord_Btm().at(0, 0);

			const arma::mat IntrFcCoord_Cent1 		= dscrt->getIntrFcCoord_Cent().at(0, 1);
			const arma::mat IntrFcCoord_Top1		= dscrt->getIntrFcCoord_Top().at(0, 1);
			const arma::mat IntrFcCoord_Btm1		= dscrt->getIntrFcCoord_Btm().at(0, 1);

			const arma::mat Sw_IntrFcCoord_Cent1	= dscrt->getSw_IntrFcCoord_Cent().at(0, 1);
			const arma::mat Sw_IntrFcCoord_Top1		= dscrt->getSw_IntrFcCoord_Top().at(0, 1);
			const arma::mat Sw_IntrFcCoord_Btm1		= dscrt->getSw_IntrFcCoord_Btm().at(0, 1);

			// Get the coordinate of the central source point
			const double SrcIndex = ((double)config.getGeom().NumSourcePt_IntrFc_Total + 1) / 2;
			arma::mat sourceCenterBtm = IntrFcCoord_Btm0.submat(0, SrcIndex - 1, 2, SrcIndex - 1);
			sourceCenterBtm = sourceCenterBtm.st();
			
			//arma::cx_mat PR_L = arma::cx_mat(1, f.n_rows);
			//arma::cx_mat D_L = arma::cx_mat(1, f.n_rows);
			/*
			std::unique_ptr<std::vector<arma::cx_mat>> PR_L =
					std::unique_ptr<std::vector<arma::cx_mat>> (new std::vector<arma::cx_mat> (f.n_rows));
			std::unique_ptr<std::vector<arma::cx_mat>> D_L =
					std::unique_ptr<std::vector<arma::cx_mat>> (new std::vector<arma::cx_mat> (f.n_rows));
			/*

			std::vector<arma::cx_mat>* PR_L =
					new std::vector<arma::cx_mat> (f.n_rows);
			std::vector<arma::cx_mat>* D_L =
					new std::vector<arma::cx_mat> (f.n_rows);
			/*/

			//arma::mat rPR_L(1, f.n_rows);
			//arma::mat iPR_L(1, f.n_rows);
			arma::cx_mat PR_L(1, f.n_rows);
			arma::cx_mat D_L(1, f.n_rows);

			//arma::field<arma::cx_mat>* PR_L = new arma::field<arma::cx_mat>(1, f.n_rows);
			//arma::field<arma::cx_mat>* D_L = new arma::field<arma::cx_mat>(1, f.n_rows);

			//arma::mat<arma::cx_mat>* PR_L = new arma::mat<arma::cx_mat>(1, f.n_rows);
			//arma::mat<arma::cx_mat>* D_L = new arma::mat<arma::cx_mat>(1, f.n_rows);

			for (ul j = 0; j < f.n_rows; j++) {
				int globalIdx = (frequencies * i) + j;
				const std::string programStr = std::to_string(globalIdx) + "\n";

				std::cout << "\n \n \n **Starting frequency" + std::to_string(globalIdx) + " program**\n";

				vat::CxMat<F>* gMat = new vat::CxMat<F>(cache, gmWidth, gmWidth, config.maxBlockSize);
				gMat->fill(std::complex<F>(0, 0));
				
				std::cout << "Populating subviews for " + programStr;

				// Row chunk #1
				vat::CxMat<F>* vMSS  = gMat->subView(0lu, 0lu, b1 - 1lu, b1 - 1lu);
				vat::CxMat<F>* vMSi  = gMat->subView(0lu,  b1, b1 - 1lu, b2 - 1lu);

				// Row chunk #2
				vat::CxMat<F>* vDF31S = gMat->subView( b1, 0lu, b2 - 1lu, b1 - 1lu);
				vat::CxMat<F>* vDF311 = gMat->subView( b1,  b1, b2 - 1lu, b2 - 1lu);
				vat::CxMat<F>* vDS3110 = gMat->subView(b1, b2, b2 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vDS3111 = gMat->subView(b1, b3, b2 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vDS3112 = gMat->subView(b1, b4, b2 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vDS3120 = gMat->subView(b1, b5, b2 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vDS3121 = gMat->subView(b1, b6, b2 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vDS3122 = gMat->subView(b1, b7, b2 - 1lu, b8 - 1lu);

				// Row chunk #3
				vat::CxMat<F>* vQiS  = gMat->subView( b2, 0lu, b3 - 1lu, b1 - 1lu);
				vat::CxMat<F>* vQ11  = gMat->subView( b2,  b1, b3 - 1lu, b2 - 1lu);
				vat::CxMat<F>* vS33110 = gMat->subView(b2, b2, b3 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS33111 = gMat->subView(b2, b3, b3 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS33112 = gMat->subView(b2, b4, b3 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS33120 = gMat->subView(b2, b5, b3 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS33121 = gMat->subView(b2, b6, b3 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS33122 = gMat->subView(b2, b7, b3 - 1lu, b8 - 1lu);

				// Row chunk #4
				vat::CxMat<F>* vS31110 = gMat->subView(b3, b2, b4 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS31111 = gMat->subView(b3, b3, b4 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS31112 = gMat->subView(b3, b4, b4 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS31120 = gMat->subView(b3, b5, b4 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS31121 = gMat->subView(b3, b6, b4 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS31122 = gMat->subView(b3, b7, b4 - 1lu, b8 - 1lu);

				// Row chunk #5
				vat::CxMat<F>* vS32110 = gMat->subView(b4, b2, b5 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS32111 = gMat->subView(b4, b3, b5 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS32112 = gMat->subView(b4, b4, b5 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS32120 = gMat->subView(b4, b5, b5 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS32121 = gMat->subView(b4, b6, b5 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS32122 = gMat->subView(b4, b7, b5 - 1lu, b8 - 1lu);

				// Row chunk #6
				vat::CxMat<F>* vDS3210 = gMat->subView(b5, b2, b6 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vDS3211 = gMat->subView(b5, b3, b6 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vDS3212 = gMat->subView(b5, b4, b6 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vDS3220 = gMat->subView(b5, b5, b6 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vDS3221 = gMat->subView(b5, b6, b6 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vDS3222 = gMat->subView(b5, b7, b6 - 1lu, b8 - 1lu);
				vat::CxMat<F>* vDF322 = gMat->subView(b5, b8, b6 - 1lu, b9 - 1lu);
				vat::CxMat<F>* vDF32R = gMat->subView(b5, b9, b6 - 1lu, b10 - 1lu);

				// Row chunk #7
				vat::CxMat<F>* vS33210 = gMat->subView(b6, b2, b7 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS33211 = gMat->subView(b6, b3, b7 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS33212 = gMat->subView(b6, b4, b7 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS33220 = gMat->subView(b6, b5, b7 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS33221 = gMat->subView(b6, b6, b7 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS33222 = gMat->subView(b6, b7, b7 - 1lu, b8 - 1lu);
				vat::CxMat<F>* vQ22 = gMat->subView(b6, b8, b7 - 1lu, b9 - 1lu);
				vat::CxMat<F>* vQiR = gMat->subView(b6, b9, b7 - 1lu, b10 - 1lu);

				// Row chunk #8
				vat::CxMat<F>* vS31210 = gMat->subView(b7, b2, b8 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS31211 = gMat->subView(b7, b3, b8 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS31212 = gMat->subView(b7, b4, b8 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS31220 = gMat->subView(b7, b5, b8 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS31221 = gMat->subView(b7, b6, b8 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS31222 = gMat->subView(b7, b7, b8 - 1lu, b8 - 1lu);

				// Row chunk #9
				vat::CxMat<F>* vS32210 = gMat->subView(b8, b2, b9 - 1lu, b3 - 1lu);
				vat::CxMat<F>* vS32211 = gMat->subView(b8, b3, b9 - 1lu, b4 - 1lu);
				vat::CxMat<F>* vS32212 = gMat->subView(b8, b4, b9 - 1lu, b5 - 1lu);
				vat::CxMat<F>* vS32220 = gMat->subView(b8, b5, b9 - 1lu, b6 - 1lu);
				vat::CxMat<F>* vS32221 = gMat->subView(b8, b6, b9 - 1lu, b7 - 1lu);
				vat::CxMat<F>* vS32222 = gMat->subView(b8, b7, b9 - 1lu, b8 - 1lu);

				// Row chunk #10
				vat::CxMat<F>* vMRi = gMat->subView(b9, b8, b10 - 1lu, b9 - 1lu);
				vat::CxMat<F>* vMRR = gMat->subView(b9, b9, b10 - 1lu, b10 - 1lu);

				const double AngTestPt	= config.getGeom().AngTestPt;
				const double Solid_rho	= config.getSolid().Solid_rho;
			
				double freq = f.at(globalIdx, 0);

				if (freq == 0.0) freq = 0.01;
				if (freq > 100) freq = 100;
				double w = 2 * M_PI * freq;
				double WaveNum_P = w / config.getFluid().WaveVel_P;

				std::cout << "Calculating Christofel solution for " + programStr;

				std::unique_ptr<ChristofelSphere> sC (new ChristofelSphere(AngTestPt));
				sC->solve(config.getSolid().getC(), Solid_rho, w, AngTestPt);
				
				std::cout << "Starting PressureFluidMat1 for " + programStr;

				std::unique_ptr<PressureFluidMat> pF (new PressureFluidMat(
										(unsigned long)config.getGeom().NumSourcePt_IntrFc_Total,
										(unsigned long)nTr,
										cache,
										config.maxBlockSize));

				// Replace independent view of the solid green matrix with references
				pF->replaceViews(vMSS, vQiS, vMSi, vQ11);
				pF->solve(gs.TransCoord_Cent0, gs.TransCoord_Btm0, IntrFcCoord_Cent0, IntrFcCoord_Top0,
				WaveNum_P, config.getFluid().Fluid_rho, config.getFluid().WaveVel_P, config.getTransducer().Rotation_Trans1, 0);

				std::cout << "Starting PressureFluidMat2 for " + programStr;

				std::unique_ptr<PressureFluidMat> pF1 (new PressureFluidMat(
						(unsigned long)config.getGeom().NumSourcePt_IntrFc_Total,
						(unsigned long)nTr,
						cache,
						config.maxBlockSize));

				// Replace independent view of the solid green matrix with references
				pF1->replaceViews(vMRR, vQiR, vMRi, vQ22);
				pF1->solve(gs.TransCoord_Cent1, gs.TransCoord_Top1, IntrFcCoord_Cent1, IntrFcCoord_Btm1,
				WaveNum_P, config.getFluid().Fluid_rho, config.getFluid().WaveVel_P, config.getTransducer().Rotation_Trans2, 1);

				std::cout << "PressureFluidMat passed for " + programStr;
				std::cout << "Starting DisFluidMat1 for " + programStr;

				std::unique_ptr<DisFluidMat> dF (new DisFluidMat(
						nTr,
						config.getGeom().NumSourcePt_IntrFc_Total,
						cache,
						config.maxBlockSize));

				// Replace independent view of the solid green matrix with references
				dF->replaceView(DisFluidMat::DisMat::DF3isView, vDF31S);
				dF->replaceView(DisFluidMat::DisMat::DF3iiView, vDF311);
				dF->solve(gs.TransCoord_Cent0, gs.TransCoord_Btm0, IntrFcCoord_Cent0, IntrFcCoord_Top0, WaveNum_P,
						freq, config.getFluid().Fluid_rho);
				
				std::cout << "Starting DisFluidMat2 for " + programStr;
				std::unique_ptr<DisFluidMat> dF1 (new DisFluidMat(
						config.getGeom().NumSourcePt_IntrFc_Total,
						nTr,
						cache,
						config.maxBlockSize));

				dF1->replaceView(DisFluidMat::DisMat::DF3ssView, vDF322);
				dF1->replaceView(DisFluidMat::DisMat::DF3siView, vDF32R);
				dF1->solve(IntrFcCoord_Cent1, IntrFcCoord_Btm1, gs.TransCoord_Cent1, gs.TransCoord_Top1, WaveNum_P,
						freq, config.getFluid().Fluid_rho);

				std::cout << "DisFluidMat passed for " + programStr;

				// Compared with MATLAB and DisFluidMat result matches
				/*
				 * Calling of sub-program AnisoGreen to calculate the Displacement and
				 * Stress Green's Function Matrix for Solid Interface
				 */

				// Green's Function Matrix at Interface 0 due to bottom point sources
				// Replace independent view of the solid green matrix with references

				std::cout << "Starting AnisoGreen for " + programStr;
				std::unique_ptr<AnisoGreen<F>> aG (new AnisoGreen<F>(
						config.getGeom().NumSourcePt_IntrFc_Total,
						cache,
						config.maxBlockSize));

				delete aG->cDS3[0];  delete aG->cDS3[1];  delete aG->cDS3[2];
				delete aG->cS31[0];  delete aG->cS31[1];  delete aG->cS31[2];
				delete aG->cS32[0];  delete aG->cS32[1];  delete aG->cS32[2];
				delete aG->cS33[0];  delete aG->cS33[1];  delete aG->cS33[2];

				aG->cDS3[0] = vDS3110; aG->cDS3[1] = vDS3111; aG->cDS3[2] = vDS3112;
				aG->cS31[0] = vS31110; aG->cS31[1] = vS31111; aG->cS31[2] = vS31112;
				aG->cS32[0] = vS32110; aG->cS32[1] = vS32111; aG->cS32[2] = vS32112;
				aG->cS33[0] = vS33110; aG->cS33[1] = vS33111; aG->cS33[2] = vS33112;

				//Sw_IntrFcCoord_Cent0.print();
				aG->solve(&Sw_IntrFcCoord_Cent0, &sourceCenterBtm, *sC,
							&config.getSolid().getC(),
							config.getGeom().NumSourcePt_IntrFc_x,
							config.getGeom().NumSourcePt_IntrFc_y,
							config.getSolid().Solid_rho,
							w,
							config.getGeom().AngTestPt,
							config.getGeom().AngTestPt,
							config,
							cache);

				/*
				 * These matrices don't have any utility in the moment and hold a lot of
				 * memory, so for now it is being released
				 */

				for (ul i = 0; i < 3; i++) {
					delete aG->cDS1[i];
					delete aG->cDS2[i];
					delete aG->cS11[i];
					delete aG->cS22[i];
					delete aG->cS12[i];
				}

				delete[] aG->cDS1;
				delete[] aG->cDS2;
				delete[] aG->cS11;
				delete[] aG->cS22;
				delete[] aG->cS12;

				std::cout << "Starting AnisoGreen1 for " + programStr;
				std::unique_ptr<AnisoGreen<F>> aG1 (new AnisoGreen<F>(
						config.getGeom().NumSourcePt_IntrFc_Total,
						cache,
						config.maxBlockSize));

				// Replace independent view of the solid green matrix with references
				delete aG1->cDS3[0];  delete aG1->cDS3[1];  delete aG1->cDS3[2];
				delete aG1->cS31[0];  delete aG1->cS31[1];  delete aG1->cS31[2];
				delete aG1->cS32[0];  delete aG1->cS32[1];  delete aG1->cS32[2];
				delete aG1->cS33[0];  delete aG1->cS33[1];  delete aG1->cS33[2];

				aG1->cDS3[0] = vDS3210; aG1->cDS3[1] = vDS3211; aG1->cDS3[2] = vDS3212;
				aG1->cS31[0] = vS31210; aG1->cS31[1] = vS31211; aG1->cS31[2] = vS31212;
				aG1->cS32[0] = vS32210; aG1->cS32[1] = vS32211; aG1->cS32[2] = vS32212;
				aG1->cS33[0] = vS33210; aG1->cS33[1] = vS33211; aG1->cS33[2] = vS33212;

				aG1->solve(&Sw_IntrFcCoord_Cent1, &sourceCenterBtm, *sC,
						&config.getSolid().getC(),
						config.getGeom().NumSourcePt_IntrFc_x,
						config.getGeom().NumSourcePt_IntrFc_y,
						config.getSolid().Solid_rho,
						w,
						config.getGeom().AngTestPt,
						config.getGeom().AngTestPt,
						config,
						cache);

				//aG1->save(getTimePath(timePoint, path, "", globalIdx));

				/*
				 * These matrices don't have any utility in the moment and hold a lot of
				 * memory, so for now it is being released
				 */
				for(ul i = 0; i < 3; i++) {
					delete aG1->cDS1[i];
					delete aG1->cDS2[i];
					delete aG1->cS11[i];
					delete aG1->cS22[i];
					delete aG1->cS12[i];
				}

				delete[] aG1->cDS1;
				delete[] aG1->cDS2;
				delete[] aG1->cS11;
				delete[] aG1->cS22;
				delete[] aG1->cS12;

				std::cout << "AnisoGreen passed for " + programStr;

				vDS3120->rotate(vDS3210, true);	
				vS33120->rotate(vS33210);
				vS31120->rotate(vS31210, true);
				vS32120->rotate(vS32210, true);

				vDS3220->rotate(vDS3110, true);
				vS33220->rotate(vS33110);
				vS31220->rotate(vS31110, true);
				vS32220->rotate(vS32110, true);

				vDS3121->rotate(vDS3211, true);
				vS33121->rotate(vS33211);
				vS31121->rotate(vS31211, true);
				vS32121->rotate(vS32211, true);

				vDS3221->rotate(vDS3111, true);
				vS33221->rotate(vS33111);
				vS31221->rotate(vS31111, true);
				vS32221->rotate(vS32111, true);

				vDS3122->rotate(vDS3212);
				vS33122->rotate(vS33212, true);
				vS31122->rotate(vS31212);
				vS32122->rotate(vS32212);

				vDS3222->rotate(vDS3112);
				vS33222->rotate(vS33112, true);
				vS31222->rotate(vS31112);
				vS32222->rotate(vS32112);

				std::cout << "SISMAG implementation passed for " + programStr;

				std::complex<F>* V = new std::complex<F>[gmWidth];
    			for (int a = 0; a < gmWidth; a++) {
					if (a < nTr) {
						V[a] = config.getTransducer().Vso;
					}
					else if (a >= nTr + 8 * nTo && a < 2 * nTr + 8 * nTo) {
						V[a] = config.getTransducer().Vto;
					}
					else {
						V[a] = 0;
					}
				}		

    			vat::AugmentedMatrix<std::complex<F>>* MATRIX1 = new vat::AugmentedMatrix<std::complex<F>>(gMat, V);
    			std::complex<F>* data = MATRIX1->solve(false);

				std::cout << "Starting matrix solution for " + programStr;
				arma::Mat<std::complex<F>> AS = arma::Mat<std::complex<F>> (data, gMat->getN_Rows(), 1, false);

				//MATRIX.reset();
				//V.reset();

				std::cout << "Matrix solution passed for " + programStr;

				/*
				 * Computation of the Source Strength and their distribution among different
				 * layers accordingly
				 */

				// Source Strengths for the distributed point sources of transducer
				cx_mat As(1, gs.NumSourcePt_Trans);
				for (int i = 0; i < gs.NumSourcePt_Trans; i++) {
					As(0, i) = AS.at(i, 0);
				}
				
				printf("As Implementation passed\n");
				double A1_numterms = config.getGeom().NumSourcePt_IntrFc_Total;

				// Source Strengths for the distributed point sources at the bottom of interface
				cx_mat A1(1, A1_numterms);
				for (int i = gs.NumSourcePt_Trans, p = 0; i < gs.NumSourcePt_Trans + config.getGeom().NumSourcePt_IntrFc_Total; i++, p++) {
					A1(0, p) = AS.at(i, 0);
				}

				std::cout << "A1 Implementation passed for " + programStr;

				double A_1_numterms = 3 * config.getGeom().NumSourcePt_IntrFc_Total;

				// Source Strengths for the distributed point sources at the top of interface
				cx_mat A_1(1, A_1_numterms);
				for (int i = gs.NumSourcePt_Trans + config.getGeom().NumSourcePt_IntrFc_Total, q = 0; i < gs.NumSourcePt_Trans + (4 * config.getGeom().NumSourcePt_IntrFc_Total); i++, q++) {
					A_1(0, q) = AS.at(i, 0);
				}

				std::cout << "A_1 Implementation passed for " + programStr;

				double A2_numterms = 3 * config.getGeom().NumSourcePt_IntrFc_Total;

				// Source Strengths for the distributed point sources at the top of interface
				cx_mat A2(1, A2_numterms);
				for (int i = gs.NumSourcePt_Trans + (4 * config.getGeom().NumSourcePt_IntrFc_Total), q = 0; i < gs.NumSourcePt_Trans + (7 * config.getGeom().NumSourcePt_IntrFc_Total); i++, q++) {
					A2(0, q) = AS.at(i, 0);
				}

				std::cout << "A2 Implementation passed for " + programStr;

				double A_2_numterms = config.getGeom().NumSourcePt_IntrFc_Total;

				// Source Strengths for the distributed point sources at the top of interface
				cx_mat A_2(1, A_2_numterms);
				for (int i = gs.NumSourcePt_Trans + (7 * config.getGeom().NumSourcePt_IntrFc_Total), q = 0; i < gs.NumSourcePt_Trans + (8 * config.getGeom().NumSourcePt_IntrFc_Total); i++, q++) {
					A_2(0, q) = AS.at(i, 0);
				}

				std::cout << "A_2 implementation passed for " + programStr;

				// Source Strengths for the distributed point sources of transducer
				cx_mat Ar(1, gs.NumSourcePt_Trans);
				for (int i = gs.NumSourcePt_Trans + (8 * config.getGeom().NumSourcePt_IntrFc_Total), q = 0; i < (2 * gs.NumSourcePt_Trans + 8 * config.getGeom().NumSourcePt_IntrFc_Total); i++, q++) {
					Ar(0, q) = AS.at(i, 0);
				}
				
				std::cout << "Ar implementation passed for " + programStr;

				AS.reset();

				ul nSrcX   = config.getGeom().NumSourcePt_IntrFc_x;
				ul nSrcY   = config.getGeom().NumSourcePt_IntrFc_y;
				ul nSrcZ   = config.getWavefield().NumTarget_z;
				ul nTrgX   = nSrcX;
				ul nTrgY   = nSrcY;
				ul nTrgZ   = nSrcZ;

				arma::vec nTrgVec(3);
				nTrgVec << nTrgX << nTrgY << nTrgZ;

				ul nSweepTrgX = 2 * nTrgX - 1lu;
				ul nSweepTrgY = 2 * nTrgY - 1lu;
				arma::vec nSweepTrgVec(2);
				nSweepTrgVec << nSweepTrgX << nSweepTrgY;

				arma::cx_mat A_1s;
				arma::cx_mat A2s;
				if (plotModeFound<int>(*config.plotModes, 1) || plotModeFound<int>(*config.plotModes, 3)) {
					A_1s = arma::cx_mat(3, config.getGeom().NumSourcePt_IntrFc_Total);
					A2s = arma::cx_mat(3, config.getGeom().NumSourcePt_IntrFc_Total);

					int index = 0;
					for (int j = 0; j < 3; j++) {
						for (int indice = 0; indice < config.getGeom().NumSourcePt_IntrFc_Total; indice++) {
							index = indice + j * config.getGeom().NumSourcePt_IntrFc_Total;
							A_1s(j, indice) = A_1(0, index);
							A2s(j, indice) = A2(0, index);
						}
					}
				}
				arma::cx_mat A_1s2;
				arma::cx_mat A2s2;
				if (plotModeFound<int>(*config.plotModes, 2)) {
					A_1s2 = arma::cx_mat(3, config.getGeom().NumSourcePt_IntrFc_Total);
					A2s2 = arma::cx_mat(3, config.getGeom().NumSourcePt_IntrFc_Total);
					arma::field<arma::cx_mat> A_1_cell(1, 3);
					arma::field<arma::cx_mat> A2_cell(1, 3);
					for (int cell = 0; cell < 3; cell++) {
						A_1_cell(0, cell) = arma::cx_mat(nTrgX, nTrgY);
						A2_cell(0, cell) = arma::cx_mat(nTrgX, nTrgY);
					}

					for (ul dir = 0; dir < 3; dir++) {
						for (ul i = 0; i < nTrgY; i++) {
							for (ul j = 0; j < nTrgX; j++) {
								ul index = j + i * nTrgX + dir * (nTrgX * nTrgY);
								A_1_cell(0, dir)(j, i) = A_1(0, index);
								A2_cell(0, dir)(j, i) = A2(0, index);
							}
						}
					}

					for (ul dir = 0; dir < 3; dir++) {
						for (ul i = 0; i < nTrgX; i++) {
							for (ul j = 0; j < nTrgY; j++) {
								ul index = j + i * nTrgY;
								A_1s2(dir, index) = A_1_cell(0, dir)(i, j);
								A2s2(dir, index) = A2_cell(0, dir)(i, j);
							}
						}
					}
				}

				/*
				std::vector<std::string>  SourceTags = { "As", "A1", "A_1s", "A2s", "A_2", "Ar" };
				std::vector<arma::cx_mat> SourceData = { As, A1, A_1s, A2s, A_2, Ar };
				savePlotData(3, SourceData, SourceTags, config.plotOutputFolder, config.validationFolder,
						config.validationMode, config.doublePrecision);
				/*/

				// For Pressure and Displacement Calculation at the fluid point
				//printf("Before Wave Field Calculation \n");
				// Pressure Calculation at the fluid point
				std::unique_ptr<LinePressureFluidMat> pFw (new LinePressureFluidMat(
						config.getGeom().NumSourcePt_IntrFc_Total,
						nTr,
						1,
						cache,
						config.maxBlockSize));

				pFw->solve(config.getFluid().getFluidPoint(), gs.TransCoord_Top1, IntrFcCoord_Btm1,
						WaveNum_P, config.getFluid().Fluid_rho, config.getFluid().WaveVel_P, config.getTransducer().Rotation_Trans2, 1);

				std::cout << "Pressure Calculation at fluid point is finished for " + programStr;

				//Displacement Calculation at the fluid point
				std::unique_ptr<LineDisFluidMat> dFw (new LineDisFluidMat(
						nTr,
						config.getGeom().NumSourcePt_IntrFc_Total, 1));
				//printf("before solving disfluidmat\n");

				dFw->solve(config.getFluid().getFluidPoint(), gs.TransCoord_Top1, IntrFcCoord_Btm1,
						WaveNum_P, freq, config.getFluid().Fluid_rho);

				std::cout << "Displacement calculation at fluid point is finished for " + programStr;

				// Wave Field at fluid point
				

				//PR_L(0,j) = pFw->getQls() * Ar.st() + pFw->getQli() * A_2.st();
				//D_L(0,j) = (dFw->getDF3ls()) * (Ar.st()) + (dFw->getDF3li()) * (A_2.st());

				arma::cx_mat tPR_L = arma::cx_mat(1, 1);
				arma::cx_mat tD_L = arma::cx_mat(1, 1);

				tPR_L = pFw->getQls() * Ar.st() + pFw->getQli() * A_2.st();
				tD_L = (dFw->getDF3ls()) * (Ar.st()) + (dFw->getDF3li()) * (A_2.st());


				//PR_L->at(j) = conj(tPR_L)*frAmp(j)/config.getTimedomain().NumSampPt;
				//D_L->at(j) = conj(tD_L)*frAmp(j)/config.getTimedomain().NumSampPt;

				PR_L(0,j) = conj(tPR_L(0,0))*frAmp(j)/config.getTimedomain().NumSampPt;
				D_L(0,j) = conj(tD_L(0,0))*frAmp(j)/config.getTimedomain().NumSampPt;

				std::cout << "Wave Field Calculation at fluid point finished for " + programStr;

				delete gMat;
			}

			arma::cx_mat TPR_L(1, config.getTimedomain().NumSampPt);
			arma::cx_mat TD_L(1, config.getTimedomain().NumSampPt);

			TPR_L = join_rows(PR_L,fliplr(PR_L));
			TD_L = join_rows(D_L,fliplr(D_L));

			arma::cx_mat TotPR_L = ifft(TPR_L);
			arma::cx_mat TotD_L = ifft(TD_L);

			//*
			std::vector<std::string>  ResultTags = {"PR_L", "D_L", "fPR_L", "fD_L"};
			std::vector<arma::cx_mat> ResultData = {TotPR_L, TotD_L, TPR_L, TD_L};
			savePlotData(3, ResultData, ResultTags, config.plotOutputFolder, config.validationFolder,
					config.validationMode, config.doublePrecision);
			
			/**/


			/*
			std::ofstream output_file;
			//writing pressure data
			output_file.open("./PR_L.txt");
			std::ostream_iterator<arma::cx_mat> output_iterator(output_file, "\n");
			std::copy(PR_L->begin(), PR_L->end(), output_iterator);
			output_file.close();

			//*
			//writing displacement data
			output_file.open("./D_L.txt");
			//std::ostream_iterator<arma::cx_mat> output_iterator(output_file, "\n");
			std::copy(D_L->begin(), D_L->end(), output_iterator);
			output_file.close();
			/**/

		}
	}
	
	endtime = std::chrono::system_clock::now();   
    std::chrono::duration<double> elapsed_seconds = endtime - starttime;     
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n"; 
	
	return 0;
}
