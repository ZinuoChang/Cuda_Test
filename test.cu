#include "test.h"

// CUDA kernel definition
template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest) {   
    CudaTest -> setvalue(5.111);
    printf("Device Value = %lf\n", CudaTest->getvalue());
}


template <typename Type>
__host__ __device__ void CudaClass<Type>::add(Type& x){
    value += x;
}


template <typename Type>
void test(CudaClass<Type> myClass){

    CudaClass<Type>* class_gpu;
    cudaMalloc(&class_gpu, sizeof(CudaClass<Type>));
    cudaMemcpy(class_gpu, &myClass, sizeof(CudaClass<Type>), cudaMemcpyHostToDevice);
    
    // Launching CUDA kernel with host function pointer
    myKernel<<<1, 1>>>(class_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(&myClass, class_gpu, sizeof(CudaClass<Type>), cudaMemcpyDeviceToHost);


    std::cout << "Host Value after using Kernel= " << myClass.getvalue() << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

template void test<int>(CudaClass<int>);
template void test<double>(CudaClass<double>);