#include "icp.hpp"
#include <iostream>
#include <limits>

// Helper: Transform point
Eigen::Vector3f transform_point(const Eigen::Vector3f& p, const Eigen::Matrix4f& T) {
    Eigen::Vector4f p_hom(p.x(), p.y(), p.z(), 1.0f);
    Eigen::Vector4f res = T * p_hom;
    return res.head<3>();
}

Eigen::Matrix4f icp_align(const std::vector<Point3D>& source_in, const std::vector<Point3D>& target, int max_iterations) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector3f> current_source;
    for(auto& p : source_in) current_source.emplace_back(p.x, p.y, p.z);
    
    std::vector<Eigen::Vector3f> tgt_eigen;
    for(auto& p : target) tgt_eigen.emplace_back(p.x, p.y, p.z);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 1. Find Correspondences (Naive NN)
        std::vector<std::pair<int, int>> correspondences;
        double error = 0;
        
        for (size_t i = 0; i < current_source.size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_idx = -1;
            
            for (size_t j = 0; j < tgt_eigen.size(); ++j) {
                float d = (current_source[i] - tgt_eigen[j]).squaredNorm();
                if (d < min_dist) {
                    min_dist = d;
                    best_idx = (int)j;
                }
            }
            correspondences.push_back({(int)i, best_idx});
            error += min_dist;
        }
        
        // Check convergence (error change)
        // ...

        // 2. Compute R, t (Kabsch)
        Eigen::Vector3f centroid_src(0,0,0), centroid_tgt(0,0,0);
        for(auto& pair : correspondences) {
            centroid_src += current_source[pair.first];
            centroid_tgt += tgt_eigen[pair.second];
        }
        centroid_src /= (float)correspondences.size();
        centroid_tgt /= (float)correspondences.size();
        
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for(auto& pair : correspondences) {
            H += (current_source[pair.first] - centroid_src) * (tgt_eigen[pair.second] - centroid_tgt).transpose();
        }
        
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
        
        if (R.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        
        Eigen::Vector3f t = centroid_tgt - R * centroid_src;
        
        Eigen::Matrix4f T_step = Eigen::Matrix4f::Identity();
        T_step.block<3,3>(0,0) = R;
        T_step.block<3,1>(0,3) = t;
        
        // Update accumulated transform
        T = T_step * T;
        
        // Update source points
        for(auto& p : current_source) {
            p = transform_point(p, T_step);
        }
    }
    
    return T;
}
