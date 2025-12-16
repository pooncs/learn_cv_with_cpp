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
std::vector<int> findKNearestNeighbors(const std::vector<Point3D>& cloud, int query_idx, int k) {
    std::vector<int> neighbors;
    std::vector<std::pair<float, int>> distances;
    distances.reserve(cloud.size());

    const auto& p1 = cloud[query_idx];

    for (int i = 0; i < cloud.size(); ++i) {
        if (i == query_idx) continue;
        const auto& p2 = cloud[i];
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        distances.push_back({dist_sq, i});
    }

    // Partial sort to get top k
    if (k > distances.size()) k = distances.size();
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

    for (int i = 0; i < k; ++i) {
        neighbors.push_back(distances[i].second);
    }
    
    // Add the query point itself to the neighborhood for PCA? 
    // Usually yes, or it doesn't matter much for large k. 
    // Let's add it to be robust.
    neighbors.push_back(query_idx);

    return neighbors;
}

// Compute normal using PCA
Eigen::Vector3f computeNormal(const std::vector<Point3D>& cloud, const std::vector<int>& indices) {
    if (indices.size() < 3) return Eigen::Vector3f(0, 0, 0); // Not enough points

    // 1. Compute Centroid
    Eigen::Vector3f centroid(0, 0, 0);
    for (int idx : indices) {
        centroid += Eigen::Vector3f(cloud[idx].x, cloud[idx].y, cloud[idx].z);
    }
    centroid /= static_cast<float>(indices.size());

    // 2. Compute Covariance Matrix
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int idx : indices) {
        Eigen::Vector3f p(cloud[idx].x, cloud[idx].y, cloud[idx].z);
        Eigen::Vector3f centered = p - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<float>(indices.size());

    // 3. Eigen Decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    
    // 4. Smallest eigenvalue corresponds to the normal
    // Eigenvalues are sorted in increasing order
    Eigen::Vector3f normal = solver.eigenvectors().col(0);

    return normal;
}

int main() {
    // 1. Create a simple plane of points
    std::vector<Point3D> cloud;
    for(float x = 0; x < 10; x += 1.0f) {
        for(float y = 0; y < 10; y += 1.0f) {
            float z = 0.0f;
            cloud.push_back({x, y, z});
        }
    }
    fmt::print("Created cloud with {} points.\n", cloud.size());

    // 2. Estimate Normals
    int k = 5;
    fmt::print("Estimating normals with k={}...\n", k);
    
    // Test center point
    int center_idx = cloud.size() / 2;
    std::vector<int> neighbors = findKNearestNeighbors(cloud, center_idx, k);
    Eigen::Vector3f normal = computeNormal(cloud, neighbors);
        
    fmt::print("Center Point Index: {}\n", center_idx);
    fmt::print("Center Point Normal: {:.3f}, {:.3f}, {:.3f}\n", normal.x(), normal.y(), normal.z());
    
    // Verify (should be close to 0,0,1 or 0,0,-1)
    if (std::abs(normal.z()) > 0.9) {
        fmt::print("Success: Normal is approximately vertical.\n");
    } else {
        fmt::print("Failure: Normal is not vertical.\n");
    }

    return 0;
}
