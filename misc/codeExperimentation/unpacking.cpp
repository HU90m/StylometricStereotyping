#include<iostream>
#include<fstream>
#include<iomanip>

#define VEC_SIZE 4

int main() {
    size_t idx;

    char byte_store[8];

    double vector[VEC_SIZE];

    std::ifstream myfile;
    myfile.open("output.dat", std::ios::in | std::ios::binary);
    if (myfile.is_open()) {
        for (idx = 0; idx < VEC_SIZE; ++idx) {
            myfile.read(byte_store, 8);
            vector[idx] = *( (double *) byte_store);
        }
    }
    myfile.close();

    std::cout << "Input: (";
    std::cout << std::fixed << std::setprecision(17) << vector[0];

    for (idx = 1; idx < VEC_SIZE; ++idx) {
        std::cout << ", " << vector[idx];
    }
    std::cout << ")" << std::endl;

    return 0;
}
