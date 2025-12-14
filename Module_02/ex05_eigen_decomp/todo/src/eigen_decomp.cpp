#include "eigen_decomp.hpp"

std::pair<Eigen::Vector2d, Eigen::Matrix2d> compute_pca(const Eigen::MatrixXd& points) {
    // TODO:
    // 1. Compute Mean of points (rowwise)
    // 2. Center the points (subtract mean)
    // 3. Compute Covariance Matrix: (centered * centered^T) / (N-1)
    // 4. Use Eigen::SelfAdjointEigenSolver to get eigenvectors
    
    return {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};
}
