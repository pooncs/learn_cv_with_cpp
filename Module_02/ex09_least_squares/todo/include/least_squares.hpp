#pragma once
#include <Eigen/Dense>

// Fit a plane ax + by + cz + d = 0 to 3D points
// points is 3xN
// Returns [a, b, c, d]
Eigen::Vector4d fit_plane(const Eigen::MatrixXd& points);
