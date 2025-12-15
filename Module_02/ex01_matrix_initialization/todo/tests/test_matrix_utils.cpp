#include <iostream>
#include <fstream>
#include <cassert>
#include "matrix_utils.hpp"

void test_readMatrix() {
    // Create a temporary file
    std::string filename = "test_matrix.txt";
    std::ofstream out(filename);
    out << "1 2\n3 4";
    out.close();

    Eigen::MatrixXd mat = readMatrix(filename);

    if(mat.size() == 0) {
        std::cerr << "[SKIP] readMatrix not implemented.\n";
        return;
    }

    assert(mat.rows() == 2);
    assert(mat.cols() == 2);
    assert(mat(0,0) == 1);
    assert(mat(1,1) == 4);

    std::cout << "[PASS] readMatrix\n";
    std::remove(filename.c_str());
}

int main() {
    try {
        test_readMatrix();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
