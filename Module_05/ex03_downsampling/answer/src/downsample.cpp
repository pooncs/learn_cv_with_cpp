#include "downsample.hpp"
#include <map>
#include <cmath>

struct VoxelIndex {
    int i, j, k;
    bool operator<(const VoxelIndex& other) const {
        return std::tie(i, j, k) < std::tie(other.i, other.j, other.k);
    }
};

struct PointSum {
    double x = 0, y = 0, z = 0;
    int count = 0;
    
    void add(const Point3D& p) {
        x += p.x;
        y += p.y;
        z += p.z;
        count++;
    }
    
    Point3D average() const {
        return { (float)(x/count), (float)(y/count), (float)(z/count) };
    }
};

std::vector<Point3D> voxel_grid_downsample(const std::vector<Point3D>& cloud, float voxel_size) {
    if (voxel_size <= 0) return cloud;
    
    std::map<VoxelIndex, PointSum> grid;
    
    for (const auto& p : cloud) {
        int i = (int)std::floor(p.x / voxel_size);
        int j = (int)std::floor(p.y / voxel_size);
        int k = (int)std::floor(p.z / voxel_size);
        
        grid[{i, j, k}].add(p);
    }
    
    std::vector<Point3D> result;
    result.reserve(grid.size());
    for (const auto& pair : grid) {
        result.push_back(pair.second.average());
    }
    
    return result;
}
