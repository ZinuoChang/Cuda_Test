#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>
// #include <nvfunctional>
// #include <functional>

using namespace Eigen;

template<typename Type>
using FunctionPtr = void(*)(Type);

template <typename Type>
class CudaClass {
public:
    CudaClass(int rows, int cols, int vec_num, Type v = 5): 
        _rows{rows}, 
        _cols{cols}, 
        _vec_num{vec_num}, 
        _matrix{MatrixXd::Identity(rows, cols)}, 
        _vectorMatrix{MatrixXd::Identity(cols, vec_num)}, 
        _result{MatrixXd::Identity(rows, vec_num)}, 
        value{v}
        {}

    __host__ __device__ inline void setvalue(Type x) {value = x; }

    __host__ __device__ inline Type getvalue() const{return value;}

     inline void setmatrix(MatrixXd& matrix){_matrix = matrix;}

     inline void setvectors(MatrixXd& vectorMatrix){_vectorMatrix = vectorMatrix;}

     inline void setresult(MatrixXd& result){_result = result;}

    __host__ __device__ void add(Type& x);

     MatrixXd MatrixMul(MatrixXd& matrix, MatrixXd& vectorMatrix);

     MatrixXd MatrixMul_array(MatrixXd& matrix, MatrixXd& vectorMatrix);

    void test(){
        MatrixXd result_cpu = _matrix * _vectorMatrix;

        MatrixXd result_gpu = MatrixMul(_matrix, _vectorMatrix);

        // MatrixXd result_array = MatrixMul_array(_matrix, _vectorMatrix);
    }

    // void 

protected:
    Type value;
    int _rows, _cols, _vec_num;
    MatrixXd _matrix, _vectorMatrix, _result;
};

template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest);


__global__ void MatMul(MatrixXd& matrix, MatrixXd& vectorMatrix, MatrixXd& result);


// template <typename Type>
// void test(CudaClass<Type> myClass);