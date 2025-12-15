#include "sor.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

std::vector<Point3D> remove_outliers(const std::vector<Point3D>& cloud, int k, float std_mul) {
    if (cloud.empty()) return {};
    
    std::vector<float> mean_distances(cloud.size());
    
    // 1. Compute mean distance to k neighbors for each point
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::vector<float> dists;
        for (size_t j = 0; j < cloud.size(); ++j) {
            if (i == j) continue;
            float dx = cloud[i].x - cloud[j].x;
            float dy = cloud[i].y - cloud[j].y;
            float dz = cloud[i].z - cloud[j].z;
            dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        
        // Sort and take k
        if (dists.empty()) {
            mean_distances[i] = 0;
            continue;
        }
        
        std::partial_sort(dists.begin(), dists.begin() + std::min((size_t)k, dists.size()), dists.end());
        
        float sum = 0;
        int count = std::min((int)dists.size(), k);
        for(int n=0; n<count; ++n) sum += dists[n];
        
        mean_distances[i] = (count > 0) ? sum / count : 0.0f;
    }
    
    // 2. Global stats
    double sum = std::accumulate(mean_distances.begin(), mean_distances.end(), 0.0);
    double mean = sum / mean_distances.size();
    
    double sq_sum = std::inner_product(mean_distances.begin(), mean_distances.end(), mean_distances.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / mean_distances.size() - mean * mean);
    
    double threshold = mean + std_mul * stdev;
    
    // 3. Filter
    std::vector<Point3D> result;
    for (size_t i = 0; i < cloud.size(); ++i) {
        if (mean_distances[i] <= threshold) {
            result.push_back(cloud[i]);
        }
    }
    
    return result;
}
