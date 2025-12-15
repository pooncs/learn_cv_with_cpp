#pragma once
#include <vector>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
};

struct PointNormal {
    float x, y, z;
    float nx, ny, nz;
};

// Computes normals for all points using k neighbors
std::vector<PointNormal> compute_normals(const std::vector<Point3D>& cloud, int k);
