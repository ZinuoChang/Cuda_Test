#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>
#include <chrono>
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

    __host__ __device__ inline void add(Type& x){value += x;}

    __host__ __device__ inline void multiplication(MatrixXd& matrix, MatrixXd& vectorMatrix, double* result, int row, int col){
        result[col*matrix.rows() + row] = matrix.row(row) * vectorMatrix.col(col);
    }

     inline void setmatrix(MatrixXd& matrix){_matrix = matrix;}

     inline void setvectors(MatrixXd& vectorMatrix){_vectorMatrix = vectorMatrix;}

     inline void setresult(MatrixXd& result){_result = result;}

     MatrixXd MatrixMul(MatrixXd& matrix, MatrixXd& vectorMatrix);

     MatrixXd MatrixMul_array(MatrixXd& matrix, MatrixXd& vectorMatrix);

    void test(){
        auto start = std::chrono::high_resolution_clock::now();
        MatrixXd result_cpu = _matrix * _vectorMatrix;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time Eigen: " << elapsed.count() << " seconds" << std::endl;

        // std::cout << result_cpu << std::endl;

        auto start1 = std::chrono::high_resolution_clock::now();
        MatrixXd result_gpu = MatrixMul(_matrix, _vectorMatrix);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = end1 - start1;
        std::cout << "Elapsed time Cuda: " << elapsed1.count() << " seconds" << std::endl;

        // std::cout << result_gpu << std::endl;

        auto start2 = std::chrono::high_resolution_clock::now();
        MatrixXd result_array = MatrixMul_array(_matrix, _vectorMatrix);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = end2 - start2;
        std::cout << "Elapsed time Cuda_array: " << elapsed1.count() << " seconds" << std::endl;

        // std::cout << result_array << std::endl;
    }

    MatrixXd _matrix, _vectorMatrix, _result;
    // void 

protected:
    Type value;
    int _rows, _cols, _vec_num;
    
};

// template<typename Type>
// __global__ void myKernel(CudaClass<Type>* CudaTest);


// __global__ void MatMul(MatrixXd& matrix, MatrixXd& vectorMatrix, MatrixXd& result, CudaClass* instance);


// template <typename Type>
// void test(CudaClass<Type> myClass);