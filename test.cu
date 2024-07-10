// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#include <Eigen/Dense>
// #include "/usr/local/include/eigen3/Eigen/Dense"
#include "test.h"


// CUDA kernel definition
__global__ void myKernel(FuncPtr funcPtr) {
    hostFunc(40);  // Call the function pointer in CUDA kernel
}

__host__ __device__ void hostFunc(int x) {
    printf("Host Function called with argument %d\n", x);
}


void test(FuncPtr function){

    // Launching CUDA kernel with host function pointer
    myKernel<<<1, 1>>>(hostFunc);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}