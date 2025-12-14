#include "least_squares.hpp"

Eigen::Vector4d fit_plane(const Eigen::MatrixXd& points) {
    // 1. Compute Centroid
    Eigen::Vector3d centroid = points.rowwise().mean();

    // 2. Center points
    Eigen::MatrixXd centered = points.colwise() - centroid;

    // 3. Compute SVD of centered points
    // We want to minimize n^T * (P - mean) * (P - mean)^T * n
    // Which is equivalent to finding the smallest eigenvector of the covariance matrix
    // Or smallest singular vector of centered matrix.
    // JacobiSVD with ComputeThinU | ComputeThinV
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered.transpose(), Eigen::ComputeThinV);
    
    // The normal is the last column of V (corresponding to smallest singular value)
    Eigen::Vector3d normal = svd.matrixV().col(2);

    // 4. Compute d = -n . centroid
    double d = -normal.dot(centroid);

    return Eigen::Vector4d(normal(0), normal(1), normal(2), d);
}
