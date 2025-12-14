#pragma once
#include <Eigen/Dense>

// Interpolate between two quaternions using SLERP
// t is between 0 and 1
Eigen::Quaterniond custom_slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t);
