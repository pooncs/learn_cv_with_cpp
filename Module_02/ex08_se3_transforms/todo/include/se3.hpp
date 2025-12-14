#pragma once
#include <Eigen/Dense>

// Create 4x4 SE3 matrix from Rotation and Translation
Eigen::Matrix4d create_se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

// Apply transform to a 3D point
Eigen::Vector3d apply_transform(const Eigen::Matrix4d& T, const Eigen::Vector3d& p);
