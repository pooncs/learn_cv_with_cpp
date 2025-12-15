#include <iostream>
#include "downsample.hpp"

int main() {
    // Generate a dense line
    std::vector<Point3D> cloud;
    for (int i = 0; i < 100; ++i) {
        // Points very close to each other
        cloud.push_back({i * 0.01f, 0, 0});
    }
    
    // Voxel size 0.1
    // Points 0-9 (0.00-0.09) -> voxel 0
    // Points 10-19 (0.10-0.19) -> voxel 1
    // Should result in ~10 points
    
    auto downsampled = voxel_grid_downsample(cloud, 0.1f);
    
    std::cout << "Original: " << cloud.size() << "\n";
    std::cout << "Downsampled: " << downsampled.size() << "\n";
    
    return 0;
}
