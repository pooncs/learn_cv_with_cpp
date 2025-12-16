#include <iostream>
#include <vector>
#include <algorithm>

class Matrix {
    int* data;
    size_t size;

public:
    // Constructor
    Matrix(size_t s) : size(s) {
        data = new int[size];
        std::cout << "Constructed Matrix(" << size << ")\n";
    }

    // Destructor
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

    // Copy Assignment Operator (Deep Copy)
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        delete[] data;
        size = other.size;
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy Assignment (Expensive!)\n";
        return *this;
    }

    // Task 2: Implement Move Constructor
    // TODO: Matrix(Matrix&& other) noexcept ...

    // Task 3: Implement Move Assignment Operator
    // TODO: Matrix& operator=(Matrix&& other) noexcept ...
};

Matrix create_matrix() {
    return Matrix(100);
}

int main() {
    std::cout << "--- Vector Push Back Test ---\n";
    std::vector<Matrix> vec;
    
    // Reserve to avoid re-allocations confusing the output (we focus on the push itself)
    vec.reserve(2); 

    std::cout << "Pushing temporary:\n";
    // This creates a temporary Matrix(100).
    // Without move semantics, it uses Copy Constructor to put it in the vector, then destroys the temp.
    // With move semantics, it uses Move Constructor (Cheaper).
    vec.push_back(Matrix(100));

    std::cout << "\nPushing another:\n";
    vec.push_back(create_matrix());

    return 0;
}
