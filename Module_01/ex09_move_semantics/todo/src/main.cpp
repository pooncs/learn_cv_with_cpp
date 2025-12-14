#include <iostream>
#include <vector>
#include <utility>

class BigMatrix {
    float* data;
    size_t size;
public:
    BigMatrix(size_t s) : size(s) {
        data = new float[size * size];
        std::cout << "Allocated " << size*size << " floats\n";
    }
    
    ~BigMatrix() {
        if(data) delete[] data;
        std::cout << "Deallocated\n";
    }

    // TODO: Implement Move Constructor
    // BigMatrix(BigMatrix&& other) noexcept ...

    // Disable copy
    BigMatrix(const BigMatrix&) = delete;
    BigMatrix& operator=(const BigMatrix&) = delete;
};

int main() {
    BigMatrix m1(100);
    
    std::cout << "Moving m1 to m2...\n";
    // TODO: Use std::move
    // BigMatrix m2 = ...
    
    return 0;
}
