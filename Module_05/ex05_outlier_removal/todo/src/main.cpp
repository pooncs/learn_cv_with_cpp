#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fmt/core.h>

struct Point3D {
    float x, y, z;
};

// Helper: Calculate distance squared
float distSq(const Point3D& a, const Point3D& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// TODO: Implement SOR Filter
// 1. For each point, find k nearest neighbors and compute average distance
// 2. Compute global mean and std_dev of these average distances
// 3. Keep points where avg_dist <= mean + alpha * std_dev
std::vector<Point3D> removeOutliers(const std::vector<Point3D>& cloud, int k, float alpha) {
    std::vector<Point3D> filtered_cloud;
    
    // Step 1: Compute mean distance for each point
    // Hint: You can reuse a simple brute-force KNN here
    
    // Step 2: Statistics
    
    // Step 3: Filter
    
    return filtered_cloud;
}

int main() {
    // 1. Create a clean cluster of points (e.g., box from 0,0,0 to 10,10,10)
    std::vector<Point3D> cloud;
    for(int i=0; i<100; ++i) {
        cloud.push_back({
            static_cast<float>(rand() % 10),
            static_cast<float>(rand() % 10),
            static_cast<float>(rand() % 10)
        });
    }
    
    // 2. Add outliers (far away points)
    cloud.push_back({100.0f, 100.0f, 100.0f});
    cloud.push_back({-50.0f, 50.0f, 0.0f});
    
    fmt::print("Original cloud size: {}\n", cloud.size());

    // 3. Apply Filter
    int k = 5;
    float alpha = 1.0f;
    auto inliers = removeOutliers(cloud, k, alpha);
    
    fmt::print("Filtered cloud size: {}\n", inliers.size());
    
    // Check if outliers are gone (size should be 100 or slightly less if strict)
    if (inliers.size() <= 100 && inliers.size() > 90) {
        fmt::print("Success: Outliers likely removed.\n");
    } else {
        fmt::print("Result seems unexpected.\n");
    }

    return 0;
}
