#include <iostream>
#include <cassert>
#include "arithmetic.hpp"

void test_matmul() {
    Eigen::Matrix2d A; A << 1, 0, 0, 1;
    Eigen::Matrix2d B; B << 2, 0, 0, 2;
    Eigen::Matrix2d C = mat_mul(A, B);
    assert(C(0,0) == 2 && C(1,1) == 2);
    std::cout << "[PASS] mat_mul\n";
}

void test_elementwise() {
    Eigen::Matrix2d A; A << 1, 2, 3, 4;
    Eigen::Matrix2d B; B << 2, 2, 2, 2;
    Eigen::Matrix2d C = element_wise(A, B);
    assert(C(0,0) == 2 && C(0,1) == 4);
    std::cout << "[PASS] element_wise\n";
}

void test_broadcast() {
    Eigen::MatrixXd M(2, 2); M << 1, 1, 1, 1;
    Eigen::Vector2d v; v << 1, 2;
    broadcast_add(M, v);
    // Col 0: 1+1=2, 1+2=3
    assert(M(0,0) == 2 && M(1,0) == 3);
    std::cout << "[PASS] broadcast_add\n";
}

int main() {
    test_matmul();
    test_elementwise();
    test_broadcast();
    return 0;
}
