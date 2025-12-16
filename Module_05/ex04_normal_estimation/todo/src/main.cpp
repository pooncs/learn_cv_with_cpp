#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <fmt/core.h>

struct Point3D {
    float x, y, z;
};

// Simple Brute Force KNN
// Returns indices of k nearest neighbors
std::vector<int> findKNearestNeighbors(const std::vector<Point3D>& cloud, int query_idx, int k) {
    std::vector<int> neighbors;
    // TODO: Implement brute force KNN
    // 1. Calculate distances to all other points
    // 2. Sort and pick top k
    
    // Hint: Store pairs of (distance, index)
    
    return neighbors;
}

// Compute normal using PCA on neighbors
Eigen::Vector3f computeNormal(const std::vector<Point3D>& cloud, const std::vector<int>& indices) {
    // TODO: Implement PCA Normal Estimation
    // 1. Compute Centroid
    // 2. Compute Covariance Matrix
    // 3. Eigen Decomposition (SelfAdjointEigenSolver)
    // 4. Return eigenvector corresponding to smallest eigenvalue
    
    return Eigen::Vector3f(0, 0, 1); // Placeholder
}

int main() {
    // 1. Create a simple plane of points
    std::vector<Point3D> cloud;
    for(float x = 0; x < 10; x += 1.0f) {
        for(float y = 0; y < 10; y += 1.0f) {
            // Plane z = 0 with some noise
            float z = 0.0f; // Could add random noise
            cloud.push_back({x, y, z});
        }
    }
    fmt::print("Created cloud with {} points.\n", cloud.size());

    // 2. Estimate Normals
    int k = 5;
    fmt::print("Estimating normals with k={}...\n", k);
    
    for (int i = 0; i < cloud.size(); ++i) {
        std::vector<int> neighbors = findKNearestNeighbors(cloud, i, k);
        Eigen::Vector3f normal = computeNormal(cloud, neighbors);
        
        // Check center point normal (should be roughly 0,0,1)
        if (i == cloud.size() / 2) {
             fmt::print("Center Point Normal: {:.3f}, {:.3f}, {:.3f}\n", normal.x(), normal.y(), normal.z());
        }
    }

    return 0;
}
