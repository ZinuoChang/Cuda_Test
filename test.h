#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>

typedef void (*FuncPtr)(int);

void test(FuncPtr function);

__global__ void myKernel(FuncPtr funcPtr);

__host__ __device__ void hostFunc(int x);

// template <typename Type>
// class cuda