# ----------------------- duplicated contents from the higher level CMakeLists ------------------------
cmake_minimum_required(VERSION 3.0)
enable_testing()
project(CUDA_TEST CXX C CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_definitions("SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}")

# Set the policy CMP0104 to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set the CUDA architectures you want to target
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75")

# Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_75; -expt-relaxed-constexpr; -rdc=true)

# Eigen
find_package(Eigen3 3.4.0 REQUIRED)
add_definitions(-DEIGEN_DONT_VECTORIZE)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_BINARY_DIR}/quadrature/libSpGH/for_testing")

# Add all the source files in src/gvi as a separate executable
cuda_add_executable(example test.cpp test.cu)

target_link_libraries(example cudadevrt)

