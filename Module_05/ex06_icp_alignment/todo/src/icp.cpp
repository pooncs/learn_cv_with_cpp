#include "icp.hpp"

Eigen::Matrix4f icp_align(const std::vector<Point3D>& source, const std::vector<Point3D>& target, int max_iterations) {
    // TODO:
    // Loop:
    // 1. Find NN correspondences
    // 2. Estimate R, t using Kabsch (SVD of covariance)
    // 3. Transform source points
    // 4. Update total transform
    return Eigen::Matrix4f::Identity();
}
