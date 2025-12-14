#include "matrix_utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== Task 1: Fixed-Size Initialization ===\n";
  Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  std::cout << "4x4 Identity:\n" << identity << "\n\n";

  std::cout << "=== Task 2: Dynamic-Size Initialization ===\n";
  int rows = 3, cols = 2;
  Eigen::MatrixXd randomMat = Eigen::MatrixXd::Random(rows, cols);
  std::cout << "Random " << rows << "x" << cols << ":\n" << randomMat << "\n\n";

  std::cout << "=== Task 3: Map from std::vector ===\n";
  std::vector<float> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> mappedMat(vec.data());
  std::cout << "Mapped Matrix:\n" << mappedMat << "\n\n";

  std::cout << "=== Task 4: Read from File ===\n";
  try {
    std::string path = "../../data/matrix.txt";
    Eigen::MatrixXd fileMat = readMatrix(path);
    std::cout << "Read from " << path << ":\n" << fileMat << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
