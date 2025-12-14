#include <iostream>
#include "solvers.hpp"

int main() {
    // Create an SPD matrix
    Eigen::Matrix3d A;
    A << 4, -1, 2,
        -1, 6, 0,
         2, 0, 5;
    
    Eigen::Vector3d b;
    b << 1, 2, 3;

    std::cout << "Matrix A (SPD):\n" << A << "\n\n";
    std::cout << "Vector b:\n" << b << "\n\n";

    std::cout << "=== Task 1: LLT Solver ===\n";
    Eigen::Vector3d x_llt = solve_llt(A, b);
    std::cout << "Solution (LLT): " << x_llt.transpose() << "\n";
    std::cout << "Error: " << relative_error(A, x_llt, b) << "\n\n";

    std::cout << "=== Task 2: LDLT Solver ===\n";
    Eigen::Vector3d x_ldlt = solve_ldlt(A, b);
    std::cout << "Solution (LDLT): " << x_ldlt.transpose() << "\n";
    std::cout << "Error: " << relative_error(A, x_ldlt, b) << "\n";

    return 0;
}
