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

// Implement SOR Filter
std::vector<Point3D> removeOutliers(const std::vector<Point3D>& cloud, int k, float alpha) {
    if (cloud.empty()) return {};

    std::vector<float> mean_distances;
    mean_distances.reserve(cloud.size());

    // Step 1: Compute mean distance for each point to its k nearest neighbors
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::vector<float> dists;
        dists.reserve(cloud.size());

        for (size_t j = 0; j < cloud.size(); ++j) {
            if (i == j) continue;
            dists.push_back(std::sqrt(distSq(cloud[i], cloud[j])));
        }

        // Partial sort to get k nearest
        if (k > dists.size()) k = dists.size();
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        double sum = 0;
        for (int n = 0; n < k; ++n) {
            sum += dists[n];
        }
        mean_distances.push_back(static_cast<float>(sum / k));
    }

    // Step 2: Statistics (Mean and StdDev of mean_distances)
    double global_sum = std::accumulate(mean_distances.begin(), mean_distances.end(), 0.0);
    double global_mean = global_sum / mean_distances.size();

    double sq_sum = std::inner_product(mean_distances.begin(), mean_distances.end(), mean_distances.begin(), 0.0);
    double global_stddev = std::sqrt(sq_sum / mean_distances.size() - global_mean * global_mean);

    // Step 3: Filter
    float threshold = static_cast<float>(global_mean + alpha * global_stddev);
    
    std::vector<Point3D> filtered_cloud;
    for (size_t i = 0; i < cloud.size(); ++i) {
        if (mean_distances[i] <= threshold) {
            filtered_cloud.push_back(cloud[i]);
        }
    }

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
    
    // Check if outliers are gone
    if (inliers.size() <= 100 && inliers.size() > 90) {
        fmt::print("Success: Outliers likely removed.\n");
    } else {
        fmt::print("Result seems unexpected.\n");
    }

    return 0;
}
