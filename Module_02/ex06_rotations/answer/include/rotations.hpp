#pragma once
#include <Eigen/Dense>

// Convert Euler Angles (Roll-X, Pitch-Y, Yaw-Z) to Rotation Matrix (ZYX order)
Eigen::Matrix3d euler_to_matrix(double roll, double pitch, double yaw);

// Convert Rotation Matrix to Axis-Angle
Eigen::AngleAxisd matrix_to_axis_angle(const Eigen::Matrix3d& R);
