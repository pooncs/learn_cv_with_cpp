#pragma once
#include <Eigen/Dense>
#include <utility>

/**
 * @brief Computes PCA of a point cloud.
 *
 * @param points 2xN matrix where each column is a 2D point.
 * @return std::pair<Eigen::Vector2d, Eigen::Matrix2d>
 *         first: Mean (centroid)
 *         second: Eigenvectors (columns are eigenvectors)
 */
std::pair<Eigen::Vector2d, Eigen::Matrix2d>
compute_pca(const Eigen::MatrixXd &points);
