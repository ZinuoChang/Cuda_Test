#include "test.h"

// CUDA kernel definition
template<typename Type>
__global__ void myKernel(CudaClass<Type>* CudaTest) {   
    CudaTest -> setvalue(5.111);
    printf("Device Value = %lf\n", CudaTest->getvalue());
}

__host__ __device__ void multiplication_vec(const VectorXd& matrix, const VectorXd& vectorMatrix, double* result, int row, int col, int mat_rows){
    result[col*mat_rows + row] = matrix.transpose() * vectorMatrix;
}

__global__ void MauMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num, CudaClass<double>* pointer) {   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < vec_num) {
        // Dimension of Matrices cannot be too large when converting or passing in kernel function
        pointer -> setvalue(5.111);
        // printf("Device Value = %lf\n", pointer->getvalue());
        MatrixXd mat, vec;
        Eigen::Map<MatrixXd> matrix_map(matrix, rows, cols);
        Eigen::Map<MatrixXd> vectorMatrix_map(vectorMatrix, cols, vec_num);

        // Cannot convert when the dimension is large
        // MatrixXd matrix_M = matrix_map;
        // MatrixXd vectorMatrix_M = vectorMatrix_map;

        // Calling function like this will cost MUCH time
        // pointer -> multiplication_vec(matrix_map.row(row), vectorMatrix_map.col(col), result, row, col, rows);
        // multiplication_vec(matrix_map.row(row), vectorMatrix_map.col(col), result, row, col, rows);
        result[col*rows + row] = matrix_map.row(row).dot(vectorMatrix_map.col(col));
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

__global__ void MauMul_Dynamic(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num, CudaClass<double>* pointer) {   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < vec_num) {
        // Calling function like this will cost MUCH time
        pointer -> multiplication_dynamic(matrix, vectorMatrix, pointer, result, row, col, rows, cols, vec_num);
    }
}

__global__ void Vector_Mul(const double* matrix, const double* vectorMatrix, double* result, int row, int col, int rows, int cols, int vec_num, CudaClass<double>* pointer){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols){
        result[col*rows + row] += matrix[idx*rows+row] * vectorMatrix[col*cols+idx];
    }
}


template <typename Type>
MatrixXd CudaClass<Type>::MatrixMul(MatrixXd& matrix, MatrixXd& vectorMatrix){
    MatrixXd result(matrix.rows(), vectorMatrix.cols());
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;
    CudaClass<Type>* class_gpu;

    cudaMalloc(&class_gpu, sizeof(CudaClass<Type>));
    cudaMemcpy(class_gpu, this, sizeof(CudaClass<Type>), cudaMemcpyHostToDevice);

    cudaMalloc(&matrix_gpu, sizeof(double)*matrix.size());
    cudaMalloc(&vectorMatrix_gpu, sizeof(double)*vectorMatrix.size());
    cudaMalloc(&result_gpu, sizeof(double)*matrix.rows()*vectorMatrix.cols());

    cudaMemcpy(matrix_gpu, matrix.data(), sizeof(double)*matrix.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix.data(), sizeof(double)*vectorMatrix.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(128, 128);
    dim3 threadperblock((vectorMatrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);
    MauMul<<<blockSize, threadperblock>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, matrix.rows(), matrix.cols(), vectorMatrix.cols(), class_gpu);
    cudaDeviceSynchronize();
    std::cout << "Host Value after using Kernel= " << this -> getvalue() << std::endl;
    cudaMemcpy(this, class_gpu, sizeof(CudaClass<Type>), cudaMemcpyDeviceToHost);

    std::cout << "Host Value after using Kernel= " << this -> getvalue() << std::endl;


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

template <typename Type>
MatrixXd CudaClass<Type>::MatrixMul_Dynamic(MatrixXd& matrix, MatrixXd& vectorMatrix){
    MatrixXd result(matrix.rows(), vectorMatrix.cols());
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;
    CudaClass<Type>* class_gpu;

    cudaMalloc(&class_gpu, sizeof(CudaClass<Type>));
    cudaMemcpy(class_gpu, this, sizeof(CudaClass<Type>), cudaMemcpyHostToDevice);

    cudaMalloc(&matrix_gpu, sizeof(double)*matrix.size());
    cudaMalloc(&vectorMatrix_gpu, sizeof(double)*vectorMatrix.size());
    cudaMalloc(&result_gpu, sizeof(double)*matrix.rows()*vectorMatrix.cols());

    cudaMemcpy(matrix_gpu, matrix.data(), sizeof(double)*matrix.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix.data(), sizeof(double)*vectorMatrix.size(), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(128, 128);
    dim3 threadperblock((vectorMatrix.cols() + blockSize.x - 1) / blockSize.x, (matrix.rows() + blockSize.y - 1) / blockSize.y);
    MauMul_Dynamic<<<blockSize, threadperblock>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, matrix.rows(), matrix.cols(), vectorMatrix.cols(), class_gpu);
    cudaDeviceSynchronize();
    std::cout << "Host Value after using Kernel= " << this -> getvalue() << std::endl;
    cudaMemcpy(this, class_gpu, sizeof(CudaClass<Type>), cudaMemcpyDeviceToHost);

    std::cout << "Host Value after using Kernel= " << this -> getvalue() << std::endl;


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
__host__ __device__ void CudaClass<Type>::multiplication_dynamic(const double* matrix, const double* vectorMatrix, CudaClass<Type>* pointer, double* result, int row, int col, int rows, int cols, int vec_num){
        // const double* matrix_data = matrix.data();
        // const double* vector_data = vectorMatrix.data();

        dim3 blockSize(64);
        dim3 threadperblock((cols + blockSize.x - 1) / blockSize.x);

        Vector_Mul<<<blockSize, threadperblock>>>(matrix, vectorMatrix, result, row, col, rows, cols, vec_num, pointer);

        // double sum = 0.0;
        // for (int i = 0; i < cols; ++i) {
        //     sum += matrix[i * rows + row] * vectorMatrix[col * cols + i];
        // }
        // result[col*rows + row] = sum;
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