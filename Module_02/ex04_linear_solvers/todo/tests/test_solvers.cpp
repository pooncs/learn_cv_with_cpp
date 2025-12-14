#include <iostream>
#include <cassert>
#include "solvers.hpp"

void test_llt() {
    Eigen::Matrix2d A; A << 2, -1, -1, 2;
    Eigen::Vector2d b; b << 1, 1;
    Eigen::Vector2d x = solve_llt(A, b);
    // Solution should be x = [1, 1]
    assert(std::abs(x(0) - 1.0) < 1e-4);
    assert(std::abs(x(1) - 1.0) < 1e-4);
    std::cout << "[PASS] solve_llt\n";
}

void test_error() {
    Eigen::Matrix2d A; A << 1, 0, 0, 1;
    Eigen::Vector2d b; b << 1, 1;
    Eigen::Vector2d x; x << 1, 1;
    double err = relative_error(A, x, b);
    assert(err < 1e-9);
    
    x << 2, 2;
    err = relative_error(A, x, b);
    assert(err > 0.1);
    std::cout << "[PASS] relative_error\n";
}

int main() {
    test_llt();
    test_error();
    return 0;
}
