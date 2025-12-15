#pragma once
#include <vector>
#include <tuple>

struct Point3D {
    float x, y, z;
};

// Voxel Grid Downsampling
// voxel_size: size of the voxel cube
std::vector<Point3D> voxel_grid_downsample(const std::vector<Point3D>& cloud, float voxel_size);
