#pragma once
#include <vector>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
};

// Returns 4x4 transformation matrix
Eigen::Matrix4f icp_align(const std::vector<Point3D>& source, const std::vector<Point3D>& target, int max_iterations = 10);
