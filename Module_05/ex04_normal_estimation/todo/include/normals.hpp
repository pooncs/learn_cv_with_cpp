#pragma once
#include <vector>

struct Point3D {
    float x, y, z;
};

struct PointNormal {
    float x, y, z;
    float nx, ny, nz;
};

std::vector<PointNormal> compute_normals(const std::vector<Point3D>& cloud, int k);
