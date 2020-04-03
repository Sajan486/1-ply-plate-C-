
cmake_minimum_required(VERSION 3.1)
project(DPSM)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(HDF5)
set (CMAKE_CXX_STANDARD 11)

# NVCC options; sets the target architecture of GPUs
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -G -g -O0 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_30,code=sm_30
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Specify include directories
include_directories(
    
    include
    include/vat/include
)

cuda_add_executable(
	
	DPSM

	# CUDA files
	include/cuDenseMatrix/CxMatrix.cuh
	include/cuDenseMatrix/util.cu
	include/cuDenseMatrix/util.cuh

	# DPSM files
	src/WaveFILMain.cpp
	src/AnisoGreen.cpp
	src/ChristofelSphere.cpp
	src/CompWaveFieldSolid.cpp
	src/CompPressureFluid.cpp
	src/Config.cpp
	src/CpuSolidGreen.cpp
	src/CuSolidGreen.cpp
	src/Discretizer.cpp
	src/DisFluidMat.cpp
	src/GeomSensCircl.cpp
	src/PressureFluidMat.cpp
	src/Settings.cpp
	src/Sphere.cpp
	src/Util.cpp

	# Supporting disk management files
	include/vat/Array.cpp
	include/vat/DiskCache.cpp
	include/vat/Grid.cpp
	include/vat/Matrix.cpp
	include/vat/VatTypes.cpp
	include/vat/Solve.cpp

	include/vat/CUDA/MatrixUtil.cu
	include/vat/CUDA/MatrixUtil.cuh
)

target_link_libraries(

	DPSM

	-llapack
	-lgomp
	-larmadillo
	-llapacke
	-lcudart -lcublas -lcusolver
	${OpenCV_LIBS}
)
