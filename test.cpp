#include "test.h"

// using Cudafunction = std::function

int main() {
    CudaClass<double> myClass(7.6);
    std::cout << "Initial Value = " << myClass.getvalue() << std::endl;

    myClass.setvalue(10.01);
    std::cout << "Host Value = " << myClass.getvalue() << std::endl;
    
    test(myClass);

    return 0;
}