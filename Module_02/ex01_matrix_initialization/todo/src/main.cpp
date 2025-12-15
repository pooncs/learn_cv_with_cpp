#include "matrix_utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== Task 1: Fixed-Size Initialization ===\n";
  // TODO: Create a 4x4 Identity matrix 'identity'
  // Eigen::Matrix4f identity = ...
  // std::cout << "4x4 Identity:\n" << identity << "\n\n";

  std::cout << "=== Task 2: Dynamic-Size Initialization ===\n";
  // TODO: Create a 3x2 matrix 'randomMat' with random values
  // Eigen::MatrixXd randomMat = ...
  // std::cout << "Random 3x2:\n" << randomMat << "\n\n";

  std::cout << "=== Task 3: Map from std::vector ===\n";
  std::vector<float> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // TODO: Map 'vec' to a 3x3 Eigen matrix 'mappedMat' WITHOUT copying.
  // Ensure the order is correct (RowMajor vs ColMajor).
  // Eigen::Map<...> mappedMat(...);
  // std::cout << "Mapped Matrix:\n" << mappedMat << "\n\n";

  std::cout << "=== Task 4: Read from File ===\n";
  try {
    std::string path = "../../data/matrix.txt";
    Eigen::MatrixXd fileMat = readMatrix(path);
    if (fileMat.size() > 0) {
      std::cout << "Read from " << path << ":\n" << fileMat << "\n";
    } else {
      std::cout << "readMatrix returned empty matrix (not implemented yet)\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
  }

  return 0;
}
