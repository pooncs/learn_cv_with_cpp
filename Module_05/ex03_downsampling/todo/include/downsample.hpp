#pragma once
#include <vector>

struct Point3D {
    float x, y, z;
};

std::vector<Point3D> voxel_grid_downsample(const std::vector<Point3D>& cloud, float voxel_size);
