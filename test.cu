#include "test.h"

// CUDA kernel definition
template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest) {   
    CudaTest -> setvalue(5.111);
    printf("Device Value = %lf\n", CudaTest->getvalue());
}

__global__ void MauMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num) {   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < vec_num) {
        Eigen::Map<MatrixXd> matrix_M(matrix, rows, cols);
        Eigen::Map<MatrixXd> vectorMatrix_M(vectorMatrix, cols, vec_num);
        result[col*rows + row] = matrix_M.row(row) * vectorMatrix_M.col(col);
        // printf("result[%d][%d] = %lf\n", row, col, result[col*rows + row]);
    }
}

__global__ void MauMul_array(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num) {   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < vec_num) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += matrix[i * rows + row] * vectorMatrix[col * cols + i];
        }
        result[col*rows + row] = sum;
    }
}


template <typename Type>
MatrixXd CudaClass<Type>::MatrixMul(MatrixXd& matrix, MatrixXd& vectorMatrix){
    MatrixXd result(matrix.rows(), vectorMatrix.cols());
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;

    cudaMalloc(&matrix_gpu, sizeof(double)*matrix.size());
    cudaMalloc(&vectorMatrix_gpu, sizeof(double)*vectorMatrix.size());
    cudaMalloc(&result_gpu, sizeof(double)*result.size());

    cudaMemcpy(matrix_gpu, matrix.data(), sizeof(double)*matrix.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix.data(), sizeof(double)*vectorMatrix.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(128, 128);
    dim3 threadperblock((vectorMatrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);
    MauMul<<<blockSize, threadperblock>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, matrix.rows(), matrix.cols(), vectorMatrix.cols());
    cudaDeviceSynchronize();
    
    cudaMemcpy(result.data(), result_gpu, sizeof(double)*result.size(), cudaMemcpyDeviceToHost);

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

template <typename Type>
MatrixXd CudaClass<Type>::MatrixMul_array(MatrixXd& matrix, MatrixXd& vectorMatrix){
    MatrixXd result(matrix.rows(), vectorMatrix.cols());
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;

    cudaMalloc(&matrix_gpu, sizeof(double)*matrix.size());
    cudaMalloc(&vectorMatrix_gpu, sizeof(double)*vectorMatrix.size());
    cudaMalloc(&result_gpu, sizeof(double)*result.size());

    cudaMemcpy(matrix_gpu, matrix.data(), sizeof(double)*matrix.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix.data(), sizeof(double)*vectorMatrix.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(128, 128);
    dim3 threadperblock((vectorMatrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);
    MauMul_array<<<blockSize, threadperblock>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, matrix.rows(), matrix.cols(), vectorMatrix.cols());
    cudaDeviceSynchronize();
    
    cudaMemcpy(result.data(), result_gpu, sizeof(double)*result.size(), cudaMemcpyDeviceToHost);

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