#include "arithmetic.hpp"
#include <iostream>

int main() {
  std::cout << "=== Task 1: Matrix Multiplication ===\n";
  Eigen::Matrix2d A;
  A << 1, 2, 3, 4;
  Eigen::Matrix2d B;
  B << 2, 0, 0, 2;

  Eigen::Matrix2d C = mat_mul(A, B);
  std::cout << "A:\n" << A << "\nB:\n" << B << "\n";
  std::cout << "A * B:\n" << C << "\n\n";

  std::cout << "=== Task 2: Element-wise Multiplication ===\n";
  Eigen::Matrix2d D = element_wise(A, B);
  std::cout << "A .element* B:\n" << D << "\n\n";

  std::cout << "=== Task 3: Broadcasting ===\n";
  Eigen::MatrixXd M(3, 4);
  M << 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3;
  Eigen::Vector3d v;
  v << 10, 20, 30;

  std::cout << "Original M:\n" << M << "\n";
  std::cout << "Vector v:\n" << v << "\n";

  broadcast_add(M, v);

  std::cout << "M after colwise += v:\n" << M << "\n";

  return 0;
}
