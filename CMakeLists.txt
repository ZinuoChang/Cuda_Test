# ----------------------- duplicated contents from the higher level CMakeLists ------------------------
cmake_minimum_required(VERSION 3.0)
enable_testing()
project(GaussianVI CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(gvi_subdirs 
    quadrature 
    ngd
    proxgd
    gvibase
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_definitions("SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}")
# add_compile_definitions(GVI_SUBDUR_ENV=0)

# Set the policy CMP0104 to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set the CUDA architectures you want to target
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80")

# Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CUDA_ARCH "sm_86") #Change it based on the GPU
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=${CUDA_ARCH}; -std=c++17; --expt-extended-lambda; --expt-relaxed-constexpr)

# Specify the CUDA standard
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Eigen
find_package(Eigen3 3.4.0 REQUIRED)
add_definitions(-DEIGEN_DONT_VECTORIZE)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_BINARY_DIR}/quadrature/libSpGH/for_testing")

# Add all the source files in src/gvi as a separate executable
add_executable(example test.cpp test.cu)
# target_link_libraries(example ${CUDA_LIBRARIES})

