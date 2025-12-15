#include "normals.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

std::vector<PointNormal> compute_normals(const std::vector<Point3D>& cloud, int k) {
    std::vector<PointNormal> result;
    result.reserve(cloud.size());
    
    // Naive O(N^2) search
    for (size_t i = 0; i < cloud.size(); ++i) {
        // 1. Find neighbors
        std::vector<std::pair<float, int>> dists;
        for (size_t j = 0; j < cloud.size(); ++j) {
            float dx = cloud[i].x - cloud[j].x;
            float dy = cloud[i].y - cloud[j].y;
            float dz = cloud[i].z - cloud[j].z;
            dists.push_back({dx*dx + dy*dy + dz*dz, (int)j});
        }
        
        // Sort first k+1 (includes self)
        std::partial_sort(dists.begin(), dists.begin() + std::min((size_t)k + 1, dists.size()), dists.end());
        
        // 2. Compute Covariance
        Eigen::Vector3f centroid(0,0,0);
        int neighbors_count = std::min((int)dists.size(), k + 1);
        
        for (int n = 0; n < neighbors_count; ++n) {
            int idx = dists[n].second;
            centroid += Eigen::Vector3f(cloud[idx].x, cloud[idx].y, cloud[idx].z);
        }
        centroid /= (float)neighbors_count;
        
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (int n = 0; n < neighbors_count; ++n) {
            int idx = dists[n].second;
            Eigen::Vector3f p(cloud[idx].x, cloud[idx].y, cloud[idx].z);
            Eigen::Vector3f diff = p - centroid;
            cov += diff * diff.transpose();
        }
        
        // 3. Eigen Decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f normal = solver.eigenvectors().col(0); // Smallest eigenvalue
        
        result.push_back({cloud[i].x, cloud[i].y, cloud[i].z, normal.x(), normal.y(), normal.z()});
    }
    
    return result;
}
