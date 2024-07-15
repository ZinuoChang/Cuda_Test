#include "test.h"

// using Cudafunction = std::function

int main() {
    // Define dimensions
    int rows = 3000;
    int cols = 10000;
    int vec_num = 3000;

    // Generate the matrices randomly
    MatrixXd matrix = MatrixXd::Random(rows, cols);
    MatrixXd vectorMatrix(cols, vec_num);
    MatrixXd result(rows, vec_num);

    // std::cout << matrix << std::endl;

    // Randomly generate the vector of VectorXd
    std::vector<VectorXd> vectors(vec_num, VectorXd(cols));
    for (int i = 0; i < vec_num; ++i) {
        vectors[i] = VectorXd::Random(cols);
    }

    // Convert vector of vectors to a matrix
    for (int i = 0; i < vec_num; ++i) {
        vectorMatrix.col(i) = vectors[i];
    }

    CudaClass<double> myClass(rows, cols, vec_num, 7.6);
    myClass.setmatrix(matrix);
    myClass.setvectors(vectorMatrix);

    // std::cout << "Initial Value = " << myClass.getvalue() << std::endl;

    // myClass.setvalue(10.01);
    // std::cout << "Host Value = " << myClass.getvalue() << std::endl;
    
    myClass.test();
    // test(myClass);

    return 0;
}