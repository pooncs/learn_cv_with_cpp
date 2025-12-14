#include "eigen_decomp.hpp"

std::pair<Eigen::Vector2d, Eigen::Matrix2d> compute_pca(const Eigen::MatrixXd& points) {
    // points is 2xN
    int N = points.cols();
    if (N < 2) return {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()};

    // 1. Compute Mean
    Eigen::Vector2d mean = points.rowwise().mean();

    // 2. Center the data
    Eigen::MatrixXd centered = points.colwise() - mean;

    // 3. Compute Covariance Matrix
    // Cov = (1 / (N-1)) * (centered * centered^T)
    Eigen::Matrix2d cov = (centered * centered.transpose()) / double(N - 1);

    // 4. Eigen Decomposition
    // SelfAdjointEigenSolver is faster for symmetric matrices
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    
    // Eigenvectors are columns of solver.eigenvectors()
    // Eigenvalues are solver.eigenvalues() (sorted increasing)
    return {mean, solver.eigenvectors()};
}
