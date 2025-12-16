#include <iostream>
#include <vector>
#include <algorithm>

class Matrix {
    int* data;
    size_t size;

public:
    Matrix(size_t s) : size(s) {
        data = new int[size];
        std::cout << "Constructed Matrix(" << size << ")\n";
    }

    ~Matrix() {
        if (data) {
            delete[] data;
            std::cout << "Destructed\n";
        }
    }

    // Copy Constructor (Deep Copy)
    Matrix(const Matrix& other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy Constructor (Expensive!)\n";
    }

    // Copy Assignment (Deep Copy)
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        delete[] data;
        size = other.size;
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy Assignment (Expensive!)\n";
        return *this;
    }

    // Task 2: Move Constructor
    // Takes an R-value reference (&&)
    Matrix(Matrix&& other) noexcept : data(nullptr), size(0) {
        // 1. Steal resources
        data = other.data;
        size = other.size;
        
        // 2. Nullify source
        other.data = nullptr;
        other.size = 0;
        
        std::cout << "Move Constructor (Cheap!)\n";
    }

    // Task 3: Move Assignment Operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) return *this;
        
        // 1. Clean up current resources
        delete[] data;
        
        // 2. Steal resources
        data = other.data;
        size = other.size;
        
        // 3. Nullify source
        other.data = nullptr;
        other.size = 0;
        
        std::cout << "Move Assignment (Cheap!)\n";
        return *this;
    }
};

Matrix create_matrix() {
    return Matrix(100);
}

int main() {
    std::cout << "--- Vector Push Back Test ---\n";
    std::vector<Matrix> vec;
    vec.reserve(2); 

    std::cout << "Pushing temporary:\n";
    vec.push_back(Matrix(100)); // Should trigger Move Constructor

    std::cout << "\nPushing another:\n";
    vec.push_back(create_matrix()); // Should trigger Move Constructor (or be elided by RVO)

    return 0;
}
