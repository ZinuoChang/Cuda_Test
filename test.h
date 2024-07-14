#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <Eigen/Dense>
// #include <nvfunctional>
// #include <functional>

template<typename Type>
using FunctionPtr = void(*)(Type);

template <typename Type>
class CudaClass {
public:
    CudaClass(Type v = 5) : value{v} {}

    __host__ __device__ inline void setvalue(Type x) {value = x; }

    __host__ __device__ inline Type getvalue() const{return value;}

    __host__ __device__ void add(Type& x);

protected:
    Type value;
};

template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest);


template <typename Type>
void test(CudaClass<Type> myClass);