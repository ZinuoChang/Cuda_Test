#include "test.h"

// CUDA kernel definition
template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest) {   
    CudaTest -> setvalue(5.111);
    printf("Device Value = %lf\n", CudaTest->getvalue());
}

__global__ void MauMul(MatrixXd& matrix, MatrixXd& vectorMatrix, MatrixXd& result) {   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%lf\n", matrix(0,0));
    // if (row < (*matrix).rows() && col < (*vectorMatrix).cols()) {
    //     (*result)(row, col) = (*matrix).row(row) * (*vectorMatrix).col(col);
    // }
}


template <typename Type>
__host__ __device__ void CudaClass<Type>::add(Type& x){
    value += x;
}

template <typename Type>
MatrixXd CudaClass<Type>::MatrixMul(MatrixXd& matrix, MatrixXd& vectorMatrix){
    MatrixXd result(matrix.rows(), vectorMatrix.cols());
    MatrixXd *matrix_gpu, *vectorMatrix_gpu, *result_gpu;
    cudaMalloc(&matrix_gpu, sizeof(MatrixXd)*matrix.rows()*matrix.cols());
    cudaMalloc(&vectorMatrix_gpu, sizeof(vectorMatrix));
    cudaMalloc(&result_gpu, sizeof(result));

    cudaMemcpy(matrix_gpu, &matrix, sizeof(MatrixXd)*matrix.rows()*matrix.cols(), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, &vectorMatrix, sizeof(vectorMatrix), cudaMemcpyHostToDevice);

    std::cout << matrix << std::endl;

    // Launch kernel
    dim3 blockSize(4, 4);
    dim3 threadperblock((vectorMatrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);
    MauMul<<<blockSize, threadperblock>>>(*matrix_gpu, *vectorMatrix_gpu, *result_gpu);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&result, result_gpu, sizeof(result), cudaMemcpyDeviceToHost);

    // std::cout << result << std::endl;

    std::cout << sizeof(vectorMatrix) << std::endl;
    std::cout << sizeof(MatrixXd) << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }


    // Free device memory
    cudaFree(matrix_gpu);
    cudaFree(vectorMatrix_gpu);
    cudaFree(result_gpu);

    return result;
}

template class CudaClass<double>;


// template <typename Type>
// void CudaClass<Type>::test(CudaClass<Type> myClass){

//     CudaClass<Type>* class_gpu;
//     cudaMalloc(&class_gpu, sizeof(CudaClass<Type>));
//     cudaMemcpy(class_gpu, &myClass, sizeof(CudaClass<Type>), cudaMemcpyHostToDevice);
    
//     // Launching CUDA kernel with host function pointer
//     myKernel<<<1, 1>>>(class_gpu);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&myClass, class_gpu, sizeof(CudaClass<Type>), cudaMemcpyDeviceToHost);


//     std::cout << "Host Value after using Kernel= " << myClass.getvalue() << std::endl;

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
//     }
// }

// template void test<int>(CudaClass<int>);
// template void test<double>(CudaClass<double>);