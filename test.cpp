#include "test.h"


// // Host function
// void hostFunc(int x) {
//     printf("Host Function called with argument %d\n", x);
// }


int main() {
    FuncPtr pointer = hostFunc;
    std::cout << "CPU:" << std::endl;
    pointer(10);
    
    test(pointer);

    return 0;
}