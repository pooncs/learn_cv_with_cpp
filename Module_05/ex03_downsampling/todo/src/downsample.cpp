#include "downsample.hpp"

std::vector<Point3D> voxel_grid_downsample(const std::vector<Point3D>& cloud, float voxel_size) {
    // TODO:
    // 1. Create a map from VoxelIndex (i,j,k) to (sum_x, sum_y, sum_z, count)
    // 2. Iterate points, compute index, accumulate
    // 3. Iterate map, compute average
    return {};
}
