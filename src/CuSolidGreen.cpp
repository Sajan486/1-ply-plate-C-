
#include "SolidGreen.h"

// Temporarily forcing the precision of SolidGreen<precision> object
template<typename F>
void cuSolidGreen(SolidGreen<double>*       sg,
				   arma::cx_mat&       aTR,
				  const arma::mat&          C,
				  ChristofelSphere&   		cSphere,
				  F                         Fluid_rho,
				  F                         freq,
				  F                         dTheta,
				  F                         dPhi,
				  F                         dispSign,
				  Config&					config) {

	printf("\tBegin Solid Green function\n");

	std::vector<int> targetDevices = config.targetDevices;
	std::mutex mtx;

	/*
	 * Because CUDA function arguments need to be accessible to every thread,
	 * minimizing the size of those arguments can go a long way to increase
	 * performance. A 4 byte unsigned int is needed at minimum to at times
	 * count up to the number of elements (<= 4,294,967,295) in a matrix.
	 * However, an 'unsigned long', though guaranteed to be at least 4 bytes,
	 * may also be larger than necessary.
	 *
	 * Moreover, an unsigned int may be 4 bytes. So we test to see which type
	 * on the current machine is just large enough to satisfy our size
	 * requirements
	 *
	 * The function floating point precision template F, is also passed as a
	 * template to create complete alias for the SolidGreen matrix, CuMatrix
	 */

	typedef unsigned int uInt;
	if (sizeof(uInt) < 4) typedef unsigned long uInt;
	using CuMatrix = cdm::CxMatrix<F, uInt>;

	uInt nX = sg->nX;
    uInt nY = sg->nY;

	freeGlobalMemory();

	// If no devices are specified, then all available devices will be used
	bool useAllAvailableDevices = (targetDevices[0] == -1);


	// Get all the available devices in the current context
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	/*
	 * If target devices are specified, make sure they exist first
	 * else just use all the devices that are found. What remains are the
	 * candidate devices
	 */

	std::map<int, double> devices;
	if (!useAllAvailableDevices) {

		/*
		 * See if there is a match with the target devices
		 */

		for (int i = 0; i < nDevices; i++) {
			for (auto device : targetDevices) {
				if (device == i) devices[i]; break;
			}
		}

	} else for (int i = 0; i < nDevices; i++) devices[i];

	/*
	 * Define the relative performance of the devices so that work can be
	 * subdivided appropriately. Best approximation of performance should
	 * probably be the product of core count and clock speed (CCS).
	 */

	cudaDeviceProp devProp;

    double totalPerformance = 0;
	for (auto device = devices.begin(); device != devices.end(); device++) {

		int deviceNumber = device->first;
		gpuErrchk(cudaGetDeviceProperties(&devProp, deviceNumber));

		long nCores      = getCoreCount(deviceNumber);
		int clockSpeed   = devProp.clockRate;
		double ccs       = (double)nCores * (double)clockSpeed;
		totalPerformance += ccs;
		device->second   = ccs;
	}

	/*
	 * See how many threads are supported in the current context. If there
	 * are not enough threads to support the number of candidate cards, the
	 * lower performance cards are removed
	 */

	int maxThreads = omp_get_max_threads();

	if (devices.size() > maxThreads) {
		printf("Warning: not enough cores to support the number of GPUs in\n"
				"the current context. Delegating work to the highest\n"
				"performance candidate cards\n");

		int nRemovals = devices.size() - maxThreads;
		for (int i = 0; i < nRemovals; i++) {

			std::map<int, double>::iterator minDevice;
			double minPerformance = devices.begin()->second;
			for (auto device = devices.begin(); device != devices.end(); device++) {
				if (device->second <= minPerformance) {
					minDevice      = device;
					minPerformance = device->second;
				}
			}
			totalPerformance -= minPerformance;
			devices.erase(minDevice);
		}
	}

	// Normalize each card's performance against the total performance
	for (auto device = devices.begin(); device != devices.end(); device++) {
		device->second /= totalPerformance;
	}


	/*
	 * Here 'nTRs' is the total number of target points that need to be processed.
	 * We then split up the input TRs among the available devices
	 */

	long nTRs = aTR.n_cols / 3lu;
	std::vector<std::tuple<uInt, uInt>> batchRanges;

	long _startIdx = 0;
	for (auto device = devices.begin(); device != devices.end(); device++) {
		long endIdx = _startIdx + (long) (device->second * (double) nTRs);
		if (endIdx >= nTRs) endIdx = nTRs - 1lu;
		batchRanges.push_back(std::make_tuple(_startIdx, endIdx));
		_startIdx = endIdx + 1;
	}
// Workload gets divided among the available GPUs here
#pragma omp parallel num_threads(devices.size())
{
	int thread = omp_get_thread_num();
	auto it = devices.begin();
	for (int i = 0; i < thread; i++) it++;

	int device = it->first;

	cudaSetDevice(device);

	std::tuple<uInt, uInt> range = batchRanges[device];
	uInt batchStartIdx = std::get<0>(range);
	uInt batchEndIdx   = std::get<1>(range);
	long batchSize = batchEndIdx - batchStartIdx + 1lu;
	arma::cx_mat deviceTR = aTR.cols(3lu * batchStartIdx, 3lu * batchEndIdx + 2lu);

	mtx.lock();
	printf("\tDevice %d target batch: %u --> %u\n", device, batchStartIdx, batchEndIdx);
	mtx.unlock();
	// tmp

    // 'img' is a handy alias for an imaginary number.
    long nTestPtSq = cSphere.getSphere().getnPts() * cSphere.getSphere().getnPts();
    std::complex<F> img(0, 1);

    // Check core count to verify optimal configuration later
    long nCores = getCoreCount(0);

    /*
     * Calculate the maximum subSize based on the available memory.
     *
     * Baseline memory is the maximum amount of memory needed before entering
     * the main loop. It varies with the number of test points.
     *
     * The memory rate is the rate at which maximum memory usage increases with
     * subSize
     *
     * Baseline memory and memory rate were related by regression against
     * nTestPt. When nTestPt was large (>=73) the regression was tight with errors no
     * greater than 5%. However, when nTestPt was smaller (<73), the
     * memory usage was more unpredictable, but did not increase beyond usage
     * where nTestPt was greater than 73.
     *
     * Therefore, for large values of nTestPt memory usage is determined by
     * regression plus a modest 15% buffer, and for small values of nTestPt,
     * memory usage was set to the memory usage at the limit of predictability.
     *
     * Max subSize is determined by subtracting the baseline memory from global
     * memory and then determining how many test points can be handled with what
     * remains.
     *
     * If the maxSubSize exceeds the number of target points, the standard
     * subSize is set to the number of target points instead
     */

    double nMBs = (double) devProp.totalGlobalMem / 1.0e6;
    double baselineMem;
    double memoryRate;
    if (cSphere.getSphere().getnPts() >= 73) {
    	baselineMem  = 0.002256 * pow(cSphere.getSphere().getnPts(), 2) -   0.01000 * cSphere.getSphere().getnPts() +  73.04;
    	memoryRate   = 0.002220 * pow(cSphere.getSphere().getnPts(), 2) - 0.0003588 * cSphere.getSphere().getnPts() + 0.1108;
    } else {
    	baselineMem  = 85.0;
    	memoryRate   = 12.0;
    }
    baselineMem *= 1.15;
    memoryRate  *= 1.15;

    double iterationMem = nMBs - baselineMem;
    long maxSubSize = (long) (iterationMem / memoryRate);

    long standardSubSize;
    if (maxSubSize > batchSize) standardSubSize = batchSize;
    else standardSubSize = maxSubSize;

    /*
     * If amount of memory is too low, the CUDA cores may not be saturated with
     * work. On average ~2.58 MB per core is needed to run efficiently, with a
     * standard deviation of ~0.25
     */

    if (nMBs / (double) nCores < 2.0) {
    	printf("\tWarning: This graphics card may not run efficiently due to\n"
    			"\tlow memory relative to the number of cores.\n");
    }

    /*
     * Given some standardSubSize, we need to determine if the total number of
     * target points can be divided up equally for processing. In most cases
     * they can't. For this reason we create a tailSubSize, which represents
     * the number of remaining target points that will be handled on the last
     * iteration. If the target points can be divided equally, uniformSubSize
     * is set to true, else it is set to false.
     */

    long nSubSections;
    long tailSubSize;
    bool uniformSubSize;

    if (batchSize % standardSubSize == 0) {
        nSubSections = batchSize / standardSubSize;
        tailSubSize = standardSubSize;
        uniformSubSize = true;
    } else {
        nSubSections = batchSize / standardSubSize + 1;
        tailSubSize = batchSize - (nSubSections - 1) * standardSubSize;
        uniformSubSize = false;
    }

    /*
     * Instantiate needed armadillo matrices. The 'a' prefix is used to
     * differentiate them from CUDA matrices.
     */

    arma::cx_mat aV   (nTestPtSq, 3ull);
    arma::cx_mat aPIJ (9ull * nTestPtSq, 3ull);
    arma::cx_mat aGAM (3ull * nTestPtSq, 1);
    arma::cx_mat fi   (3ull, 3ull);
    arma::cx_mat pij  (3ull, 3ull);

    // Reshape ThetaP and PhiP into vectors
    arma::mat ThetaP = cSphere.getSphere().getTheta().st();
    arma::mat PhiP = cSphere.getSphere().getPhi().st();
    ThetaP.reshape(arma::SizeMat(nTestPtSq, 1));
    PhiP.reshape  (arma::SizeMat(nTestPtSq, 1));

    // Convert real-valued matrices to complex ones to simplify calculations
    arma::cx_mat cosTheta   (cos(ThetaP), arma::mat(nTestPtSq, 1, arma::fill::zeros));
    arma::cx_mat sinTheta   (sin(ThetaP), arma::mat(nTestPtSq, 1, arma::fill::zeros));
    arma::cx_mat armaCosPhi (cos(PhiP),   arma::mat(nTestPtSq, 1, arma::fill::zeros));
    arma::cx_mat sinPhi     (sin(PhiP),   arma::mat(nTestPtSq, 1, arma::fill::zeros));
    ThetaP.reset();
    PhiP.reset();

    // Build armadillo matrix, V, to hold the invariant V-values.
    aV.col(0ull) = cosTheta % armaCosPhi;
    aV.col(1ull) = sinTheta % armaCosPhi;
    aV.col(2ull) = sinPhi;
    cosTheta.reset();
    sinTheta.reset();
    sinPhi.reset();

    /*
     * Iterate through each mode, theta, and phi angle to build PIJ and GAM
     * matrices. pij is a 3x3 matrix, that is appended to aPIJ on each iteration
     * aPIJ, then can be thought of as a column of 3x3 matrices. aGAM, however,
     * is a single column
     */

    for (unsigned long long l = 0; l < 3; l++) {
        for (long i = 0; i < cSphere.getSphere().getnPts(); i++) {
            for (long j = 0; j < cSphere.getSphere().getnPts(); j++) {
                long k = j + i * cSphere.getSphere().getnPts();
                long p = 3 * k + 3 * l * nTestPtSq;
                fi =  cSphere.getFI()(i, j);
                pij = fi.col(l) * trans(fi.col(l));
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        pij(m, n) = round(real(pij(m, n)) * 1e8) / 1e8;
                    }
                }
                aPIJ.row(p + 0ull) = pij.row(0ull);
                aPIJ.row(p + 1ull) = pij.row(1ull);
                aPIJ.row(p + 2ull) = pij.row(2ull);
                aGAM.row(k + l * nTestPtSq) = sqrt(cSphere.getCV()(i, j)(0ull, l));
            }
        }
    }


    /*
     * Convert armadillo matrices to CUDA matrices. Due to the benefits of
     * coalesced memory access, the target points, TR, are organized into a
     * column, leaving many operations (such as reductions) in the horizontal
     * dimension. Horizontal operations are preferred here because the CUDA
     * matrices are in the row-major order.
     */

    CuMatrix* TR = new CuMatrix(deviceTR);
    TR->transpose();

    CuMatrix cosPhi(armaCosPhi);
    armaCosPhi.reset();
    cosPhi.deckExpansion(cdm::Rows, 3);

    CuMatrix* V = new CuMatrix(aV);
    aV.reset();

    CuMatrix* GAM = new CuMatrix(aGAM);
    aGAM.reset();

    /*
     * Define all iteration-invariant matrices. These matrices are the same for
     * each target point, and therefore should only be calculated one time.
     * The most important are:
     *
     * invarV
     * invarUU1
     * invarSoln2
     * invarUU2
     * invarDispDW
     *
     * They remain at all times in global memory to ensure fast access. Many of
     * them are expanded with operations like deckExpansion. These operations
     * basically copy the original matrix and the append N duplicates to the
     * original.
     *
     * This expansion was originally done in the main loop with the hope of
     * reducing the amount of global memory occupancy. However, with profiling,
     * it was found that these expansions were expensive, and that it was a
     * better trade-off to keep the fully expanded matrices at all times.
     *
     * invarSoln2 is an exception in this case because it is only needed in a
     * small fractions of iterations as will be shown later.
     *
     * Also note that cardExpansions are used to 'inflate' matrices by
     * duplicating internal elements. This allows for one-to-one matching of
     * matrices for element-wise multiplication.
     *
     * As mentioned before, it was found that keeping matrices extended
     * in the row-major dimension enhanced memory access. This was discovered
     * after the first draft, which is why many of the matrices are transposed.
     * The overheard of building the invariant matrices is very minimal, so it
     * was seen as unnecessary to optimize before the main loop.
     *
     * Post multiplication by PIJ invariant matrices are 'unpackaged' in a
     * unique operation to transform the structure of invariants from a matrix
     * of 3x3 'boxes' to a matrix of 9x1 strips. This again further simplified
     * main loop  operations, especially for reductions where without
     * unpackaging the matrices would need to be reshaped, or reduced without
     * coalesced access.
     */

    CuMatrix* invGam = GAM->powerAsCopy(-1);
    GAM->freeMem();
    delete GAM;

    CuMatrix* cosPhiInvGam = invGam->powerAsCopy(3);
    cosPhiInvGam->multiply(&cosPhi);

    CuMatrix PIJ(aPIJ);
    aPIJ.reset();

    CuMatrix* invarV = V->copy();
    invarV->transpose();
    invarV->deckExpansion(cdm::Rows, standardSubSize);

    CuMatrix* invarSoln1 = new CuMatrix(3 * nTestPtSq, 1, std::complex<F>(dTheta * dPhi, 0));
    invarSoln1->multiply(cosPhiInvGam);
    invarSoln1->cardExpansion(cdm::Columns, 3);
    invarSoln1->cardExpansion(cdm::Rows, 3);
    invarSoln1->multiply(&PIJ);

    CuMatrix* invarUU1 = invarSoln1->multiplyAsCopy((img * freq) / (F) (8 * (pow(M_PI, 2)) * Fluid_rho));

    invarSoln1->freeMem();
    delete invarSoln1;

    invarUU1->transpose();
    invarUU1->unpackage();
    invarUU1->deckExpansion(cdm::Rows, standardSubSize);

    CuMatrix* invarSoln2 = invGam->powerAsCopy(2);
    invarSoln2->multiply(std::complex<F>(dPhi, 0));
    invarSoln2->cardExpansion(cdm::Columns, 3);
    invarSoln2->cardExpansion(cdm::Rows, 3);
    invarSoln2->multiply(&PIJ);
    invarSoln2->transpose();
    CuMatrix* invarUU2 = invarSoln2->multiplyAsCopy(std::complex<F>(1 / (8 * (pow(M_PI, 2)) * Fluid_rho), 0));
    invarUU2->unpackage();
    invarSoln2->unpackage();
    invarSoln2->multiply(-1 / (8 * (pow(M_PI, 2)) * Fluid_rho));

    CuMatrix* invarDispDW = new CuMatrix(3 * nTestPtSq, 1, std::complex<F>(dTheta * dPhi, 0));
    invarDispDW->multiply(&cosPhi);
    cosPhi.freeMem();
    CuMatrix* invGamPow5 = invGam->powerAsCopy(5);
    invarDispDW->multiply(invGamPow5);

    invGamPow5->freeMem();
    delete invGamPow5;

    invarDispDW->cardExpansion(cdm::Columns, 3);
    invarDispDW->cardExpansion(cdm::Rows, 3);
    invarDispDW->multiply(&PIJ);
    invarDispDW->multiply(-(pow(freq, 2)) / (8 * (pow(M_PI, 2))*Fluid_rho));
    PIJ.freeMem();
    invarDispDW->transpose();
    invarDispDW->unpackage();
    invarDispDW->deckExpansion(cdm::Rows, standardSubSize);

    V->cardExpansion(cdm::Columns, 3);
    V->cardExpansion(cdm::Rows, 3);
    V->deckExpansion(cdm::Rows, 3);
    V->transpose();
    V->unpackage();
    CuMatrix* vCol1 = V->rowRange( 0,  8);
    CuMatrix* vCol2 = V->rowRange( 9, 17);
    CuMatrix* vCol3 = V->rowRange(18, 26);

    V->freeMem();
    delete V;

    invGam->transpose();

    /*
     * Define outer scoped matrices.
     *
     * These matrices are the product of the main loop and are appended to as
     * the main loop progresses
     */
    CuMatrix* DGF;
    CuMatrix* UDW1;
    CuMatrix* UDW2;
    CuMatrix* UDW3;

    /*
     * Start a clock to keep track of how much time each loop takes to complete.
     * From this, we can derive an estimated time of completion.
     */

    time_t clockStart;
    double duration;
    if (device == 0) clockStart = time(NULL);

    /*
     * Main loop.
     *
     * This is where the bulk of the time is spent processing. It iterates
     * through the number of subsections in the target points, appending to the
     * target matrices UDW1, UDW2, UDW3, and DGF.
     *
     * We bring 'subN' and 'standardSection' into the outer scope so that they
     * may stay constant until a change occurs
     */

    long subN;
    bool standardSection;
    for (long i = 0; i < nSubSections; i++) {

    	/*
    	 * If this is the first loop, set the loop's subSize, subN. We reset
    	 * the accumulated run time of CUDA functions to zero, so that if we
    	 * are profiling them we negate the accumulated time that occurred
    	 * during setup.
    	 *
    	 * The first loop is always a standard section, that is, it is always a
    	 * standard subSize.
    	 *
    	 * variables subN and standardSection only change if the target point
    	 * count is not divisible by the standardSubSize and it is the last
    	 * iteration of the main loop.
    	 */

    	if (i == 0) {
    		subN = standardSubSize;
    		resetCudaTimes();
    		standardSection = true;
    	} else if (i == nSubSections - 1 && !uniformSubSize) {
    		subN = tailSubSize;
    		standardSection = false;
    	}

    	// Extract this iteration's tr values.
    	long r0 = 3 * standardSubSize * i;
    	long rf = r0 + 3 * subN - 1;
    	CuMatrix* tr = TR->submatAsCopy(r0, 0, rf, 0);

    	/*
    	 * Multiply tr values by the invariant V values. If it is a non-standard
    	 * section, we truncate V before multiplication
    	 */

    	CuMatrix* vTR;
    	if (standardSection) {
    		vTR = invarV->multiplyAsCopy(tr);
    	} else {
    		vTR = invarV->submatAsCopy(0, 0, 3 * subN - 1, nTestPtSq - 1);
    		vTR->multiply(tr);
    	}

    	/*
    	 * Give vTR the alias of interface area, as it will soon become. Card
    	 * reductions sum groups of adjacent Rows, to a single Row, and card
    	 * expansions expand a single row to a family of adjacent copies of the
    	 * original. This is the equivalent of performing a dot product of every
    	 * v and TR value and then expanding the result three times.
    	 *
    	 * Of course there is one other operation where we do a globalPredicate.
    	 * Here we essentially ask: 'are any of the dot products zero?' The
    	 * answer, anecdotally, is that iArea is completely non-zero > 99% of
    	 * the time.
    	 *
    	 * Knowing that iArea is not equal to zero for all elements of the
    	 * current iteration is valuable because it allows us to negate many
    	 * operations. That is because many other matrices can only be non-zero
    	 * if iArea is zero for some element.
    	 *
    	 * That is not to say that a zero valued iArea value will always imply
    	 * a non-zero product, but that the only time we CAN find a non-zero
    	 * element for many matrices is when for some element iArea is zero
    	 *
    	 * the variable 'zeroValueIA' is used throughout the main loop to save
    	 * computation.
    	 */

    	CuMatrix* iArea = vTR->clone();
    	iArea->cardReduction(cdm::Columns, 3);
    	bool zeroValueIA = !iArea->globalPredicate(NotEqualTo, 0);
    	iArea->deckExpansion(cdm::Columns, 3);

    	// Build matrices only if iArea is equal to zero for any element
    	CuMatrix* trCols;
    	CuMatrix* invMag;
    	if (zeroValueIA) {
    		trCols = tr->reshapeAsCopy(subN, 3, cdm::Rows, cdm::Columns);
    		trCols->reshape(3 * subN, 1, cdm::Rows, cdm::Rows);
    		trCols->cardExpansion(cdm::Rows, 9);

    		/*
    		 * Calculate the vector magnitude of the current iteration's tr
    		 * values. The result is inversed as the magnitudes are used in
    		 * division
    		 */

    		invMag = tr->clone();
    		invMag->power(2);
    		invMag->cardReduction(cdm::Columns, 3);
    		invMag->power(-1);
    		invMag->realSqrt();
    		invMag->cardExpansion(cdm::Rows, 9);
    	}

    	/*
    	 * Use the current iArea to multiply against the subSection-sized
    	 * invariant UU1 and build the iteration specific UU1 matrix
    	 */

    	CuMatrix* iAreaExp = iArea->multiplyAsCopy(img * freq);
    	iAreaExp->multiply(invGam);
    	iAreaExp->exp();
    	iAreaExp->cardExpansion(cdm::Rows, 9);
    	CuMatrix* UU1;
    	if (standardSection) {
    		UU1 = invarUU1->multiplyAsCopy(iAreaExp);
    	} else {
    		UU1 = invarUU1->submatAsCopy(0, 0, 9 * subN - 1, 3 * nTestPtSq - 1);
    		UU1->multiply(iAreaExp);
    		iAreaExp->freeMem();
    	}
    	iAreaExp->freeMem();

    	/*
    	 * Use the current iArea to multiply against the subSection-sized
    	 * invariant dispDW and build the iteration specific dispDW matrix
    	 *
    	 * DispDW will branch into three variants:
    	 *
    	 * dispDW1
    	 * dispDW2
    	 * dispDW3
    	 *
    	 * when multiplied by the x, y and z values of v, but for now operations
    	 *  are redundant
    	 */

    	iAreaExp = iArea->multiplyAsCopy(img * freq);
    	iAreaExp->exp();
    	iAreaExp->cardExpansion(cdm::Rows, 9);
    	CuMatrix* dispDW;
    	if (standardSection) {
    		dispDW = invarDispDW->multiplyAsCopy(iAreaExp);
    	} else {
    		dispDW = invarDispDW->rowRange(0, 9 * subN - 1);
    		dispDW->multiply(iAreaExp);
    	}
    	iAreaExp->freeMem();
    	delete iAreaExp;

    	/*
    	 * Set target matrices UU1 and dispDW equal to zero where
    	 * iArea is less than zero, then sum UU1 across all cdm::Rows
    	 */

    	iArea->cardExpansion(cdm::Rows, 9);
    	std::vector<CuMatrix*> targetMats = {UU1, dispDW};
    	iArea->tiledPredicate(LessThan, 0, targetMats, 0);
    	UU1->reduceBy(cdm::Rows);

    	/*
    	 * understanding when UU2 is non-zero:
    	 *
    	 * UU2 values (are there zero-valued elements?)
    	 * Zero    = (iArea != 0 || (UU1 || UU2) == 0) for all elements
    	 * NonZero = (iArea == 0 && (UU1 || UU2) != 0) for any elements
    	 *
    	 * As you can see: UU2 can only be non-zero where iArea has some element
    	 * equal to zero. Therefore: we test to see if iArea is non-zero valued.
    	 * If not, we build the UU2 matrix and add it to UU1.
    	 *
    	 * Else, we skip this step because adding all zeros does nothing to UU1
    	 */

    	if (zeroValueIA) {
    		CuMatrix* UU2 = invarUU2->deckExpansionAsCopy(cdm::Rows, subN);
    		targetMats = {UU2};
    		iArea->tiledPredicate(NotEqualTo, 0, targetMats, 0);
    		UU2->multiply(invMag);
    		UU2->reduceBy(cdm::Rows);
    		UU1->add(UU2);
    		UU2->freeMem();
    		delete UU2;
    	}

    	/*
    	 * Append the sum of UU1 and UU2 to DGF, and free the copy of UU1 in the
    	 * loop.
    	 */

    	if (i == 0) DGF = UU1->copy();
    	else DGF->join(UU1, cdm::Rows);
    	UU1->freeMem();
    	delete UU1;

    	/*
    	 * Branch dispDW for variations found by multiplying by the various
    	 * dimensions of v, then reduce. The original source dispDW matrix is
    	 * cleared.
    	 */

    	CuMatrix* dispDW2 = dispDW->multiplyAsCopy(vCol2);
    	dispDW2->reduceBy(cdm::Rows);
    	CuMatrix* dispDW3 = dispDW->multiplyAsCopy(vCol3);
    	dispDW3->reduceBy(cdm::Rows);
    	dispDW->multiply(vCol1);
    	dispDW->reduceBy(cdm::Rows);
    	dispDW->join(dispDW2, cdm::Rows);
    	dispDW->join(dispDW3, cdm::Rows);
    	dispDW2->freeMem();
    	delete dispDW2;
    	dispDW3->freeMem();
    	delete dispDW3;

    	/*
    	 * like with UU2, soln2 can only be non-zero when iArea has zero valued
    	 * elements. Therefore, we only calculate and add soln2 to UDW when
    	 * zero valued iArea elements are found.  Else, UDW is just made as
    	 * a clone of dispDW.
    	 */

    	CuMatrix* UDW;
    	if (!zeroValueIA) {
    		iArea->freeMem();
    		delete iArea;

    		UDW = dispDW->clone();
    	} else {
    		CuMatrix* soln2 = invarSoln2->deckExpansionAsCopy(cdm::Rows, subN);
    		targetMats = {soln2};
    		iArea->tiledPredicate(NotEqualTo, 0, targetMats, 0);
    		iArea->freeMem();
    		delete iArea;

    		soln2->reduceBy(cdm::Rows);

    		UDW = soln2->clone();
    		invMag->power(3);
    		UDW->multiply(invMag);
    		invMag->freeMem();
    		UDW->deckExpansion(cdm::Rows, 3);
    		UDW->multiply(trCols);
    		trCols->freeMem();
    		delete trCols;
    		UDW->add(dispDW);
    		dispDW->freeMem();
    		delete dispDW;
    	}

    	/*
    	 * Parse out the subsections of UDW and use them to append to the outer
    	 * scoped matrices UDW1, UDW2, UDW3
    	 */

    	CuMatrix* udw1 = UDW->rowRange( 0 * subN,  9 * subN - 1);
    	CuMatrix* udw2 = UDW->rowRange( 9 * subN, 18 * subN - 1);
    	CuMatrix* udw3 = UDW->rowRange(18 * subN, 27 * subN - 1);
    	UDW->freeMem();
    	delete UDW;

    	if (i == 0) {

    		UDW1 = udw1->copy();
    		UDW2 = udw2->copy();
    		UDW3 = udw3->copy();
    	} else {

    		UDW1->join(udw1, cdm::Rows);
    		UDW2->join(udw2, cdm::Rows);
    		UDW3->join(udw3, cdm::Rows);
    	}
    	udw1->freeMem();
    	delete udw1;
    	udw2->freeMem();
    	delete udw2;
    	udw3->freeMem();
    	delete udw3;

    	/*
    	 * Use the total main loop time and the current iteration to figure out
    	 * the average time per iteration. Use the average time per iteration to
    	 * determine how much time is left with the number of remaining
    	 * sub-sections.
    	 */

        long itersPerPrint = 5;
        if (i != 0 && i % itersPerPrint == 0 && device == 0) {
        	time_t clockEnd = time(NULL);
            duration = (double)(clockEnd - clockStart);
            double iterDuration = (duration) / ((double) i + 1);
            long remainingIters = nSubSections - i - 1;
            double remainingTime = remainingIters * iterDuration;
            double remainingTimeHr = floor(remainingTime / 3600.0);
            double remainingTimeMin = floor(remainingTime / 60.0) - 60.0 * remainingTimeHr;
            double remainingTimeSec = remainingTime - 3600.0 * remainingTimeHr - 60.0 * remainingTimeMin;
            double percentComplete = 100 * (double) i / nSubSections;
            if (remainingTimeHr > 0) {
            	printf("\tRemaining Time: %.0f hrs, %.0f min (%.2lf%% complete)\n", remainingTimeHr, remainingTimeMin, percentComplete);
            } else {
            	printf("\tRemaining Time: %.0f min, %.0f sec (%.2lf%% complete)\n", remainingTimeMin, remainingTimeSec, percentComplete);
            }
        }
        tr->freeMem();
        delete tr;
    }
    if (thread == 0) {
    	printf("\tRemaining Time: %.0f min, %.0f sec (%.2lf%% complete)\n", 0.0, 0.0, 100.0);
    }

    // Free memory for invariant matrices
    TR->freeMem();
    delete TR;

    invGam->freeMem();
    delete invGam;

    cosPhiInvGam->freeMem();
    delete cosPhiInvGam;

    invarV->freeMem();
    delete invarV;

    invarUU1->freeMem();
    delete invarUU1;

    invarSoln2->freeMem();
    delete invarSoln2;

    invarUU2->freeMem();
    delete invarUU2;

    invarDispDW->freeMem();
    delete invarDispDW;

    vCol1->freeMem();
    delete vCol1;

    vCol2->freeMem();
    delete vCol2;

    vCol3->freeMem();
    delete vCol3;

    /*
     * Return output matrices to a format more congruent with the rest of the
     * program
     */

    DGF->package();
    UDW1->package();
    UDW2->package();
    UDW3->package();

    // Arrange UDW matrices into a row of N strain matrices.
    CuMatrix* upperEPS;
    CuMatrix* eps0 = UDW1->submatAsCopy(0, 0, 3 * batchSize - 1, 0);
    CuMatrix* eps1 = UDW2->submatAsCopy(0, 1, 3 * batchSize - 1, 1);
    CuMatrix* eps2 = UDW3->submatAsCopy(0, 2, 3 * batchSize - 1, 2);
    upperEPS = eps0->copy();
    upperEPS->join(eps1, cdm::Columns);
    upperEPS->join(eps2, cdm::Columns);
    eps0->freeMem();
    delete eps0;
    eps1->freeMem();
    delete eps1;
    eps2->freeMem();
    delete eps2;
    CuMatrix* lowerEPS;
    CuMatrix* UDW31 = UDW3->submatAsCopy(0, 1, 3 * batchSize - 1, 1);
    CuMatrix* UDW22 = UDW2->submatAsCopy(0, 2, 3 * batchSize - 1, 2);
    CuMatrix* eps3 = UDW31->addAsCopy(UDW22);
    UDW31->freeMem();
    delete UDW31;
    UDW22->freeMem();
    delete UDW22;
    CuMatrix* UDW12 = UDW1->submatAsCopy(0, 2, 3 * batchSize - 1, 2);
    CuMatrix* UDW30 = UDW3->submatAsCopy(0, 0, 3 * batchSize - 1, 0);
    CuMatrix* eps4 = UDW12->addAsCopy(UDW30);
    UDW12->freeMem();
    delete UDW12;
    UDW30->freeMem();
    delete UDW30;
    CuMatrix* UDW11 = UDW1->submatAsCopy(0, 1, 3 * batchSize - 1, 1);
    CuMatrix* UDW20 = UDW2->submatAsCopy(0, 0, 3 * batchSize - 1, 0);
    CuMatrix* eps5 = UDW11->addAsCopy(UDW20);
    UDW11->freeMem();
    delete UDW11;
    UDW20->freeMem();
    delete UDW20;

    UDW1->freeMem();
    delete UDW1;

    UDW2->freeMem();
    delete UDW2;

    UDW3->freeMem();
    delete UDW3;
    lowerEPS = eps3->copy();
    lowerEPS->join(eps4, cdm::Columns);
    lowerEPS->join(eps5, cdm::Columns);
    eps3->freeMem();
    delete eps3;
    eps4->freeMem();
    delete eps4;
    eps5->freeMem();
    delete eps5;
    lowerEPS->multiply(std::complex<F>(0.5, 0));
    CuMatrix* EPS = upperEPS->copy();
    EPS->join(lowerEPS, cdm::Columns);
    upperEPS->freeMem();
    delete upperEPS;
    lowerEPS->freeMem();
    delete lowerEPS;

    // Convert CUDA matrices back to armadillo matrices
    arma::cx_mat* aEPS = EPS->toArmaMat();
    arma::cx_mat* aDGF = DGF->toArmaMat();
    EPS->freeMem();
    delete EPS;
    DGF->freeMem();
    delete DGF;

    /*
     * For each strain matrix in aEPS, do a matrix multiplication with the
     * material property matrix C
     */

    arma::cx_mat SIG(3ull * batchSize, 6ull);
    for (uInt i = 0; i < 3 * batchSize; i++) {
    	arma::cx_mat strain = aEPS->submat(i, 0, i, 5).st();
    	arma::cx_mat stress = C * strain;
    	stress = stress.st();
    	SIG.submat(i, 0, i, 5) = stress;
    }
    delete aEPS;
    //aEPS->reset();


    /*
     * Package the armadillo stress matrices into the output matrices
     * each threads writes to an independent set of addresses in the output
     * matrices
     */

    uInt endIdx = std::get<1>(batchRanges[device]);
    uInt startIdx = endIdx - batchSize + 1l;

    uInt n;
    for (uInt gIdx = startIdx, idx = 0lu; gIdx <= endIdx; gIdx++, idx++) {
    	for (uInt k = 0lu; k < 3lu; k++) {

    		uInt y = gIdx / nX;
    		uInt x = gIdx - y * nX;

    		n = x + y * nX;

    		for (int i = 0; i < 3; i++) {
    			/*
    			 * NOTE: displacement signs reversed here!! I assume that this
    			 * sign is arbitrary, and it helps to fill in the solid green
    			 * matrix efficiently by performing this operation now rather
    			 * than later
    			 */
    			if (dispSign < 0) {
    				sg->data[i + 0]->at(x, y, k) = - aDGF->at(3 * idx + k, i);
    			} else {
    				sg->data[i + 0]->at(x, y, k) = aDGF->at(3 * idx + k, i);
    			}
    		}
    		for (int i = 0; i < 6; i++) {
    			sg->data[i + 3]->at(x, y, k) = SIG(3 * idx + k, i);
    		}
    	}
    }

    delete aDGF;
    SIG.reset();
}
	printf("\tEnd Solid Green function\n");
}

// Temporarily forcing the precision of SolidGreen<precision> object
template void cuSolidGreen<float> (SolidGreen<double>*,  arma::cx_mat&, const arma::mat&, ChristofelSphere&, float,  float,  float,  float, float, Config&);
template void cuSolidGreen<double>(SolidGreen<double>*,  arma::cx_mat&, const arma::mat&, ChristofelSphere&, double, double, double, double, double, Config&);
