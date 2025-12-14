#pragma once
#include <Eigen/Dense>
#include <string>

/**
 * @brief Reads a matrix from a text file.
 * File format: Space-separated values, newlines denote rows.
 * 
 * @param filename Path to the file
 * @return Eigen::MatrixXd The loaded matrix
 * @throws std::runtime_error if file cannot be opened or dimensions are inconsistent.
 */
Eigen::MatrixXd readMatrix(const std::string& filename);
