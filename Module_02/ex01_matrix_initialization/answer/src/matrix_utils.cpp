#include "matrix_utils.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

Eigen::MatrixXd readMatrix(const std::string& filename) {
    std::vector<double> values;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    double val;
    int rows = 0;
    int cols = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int current_cols = 0;
        while (ss >> val) {
            values.push_back(val);
            current_cols++;
        }
        if (cols == 0) cols = current_cols;
        else if (cols != current_cols) throw std::runtime_error("Inconsistent columns");
        rows++;
    }

    if (rows == 0) return Eigen::MatrixXd();

    // Map the vector to a matrix (RowMajor because we read line by line)
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
}
