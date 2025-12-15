#include "mahalanobis.h"

double computeMahalanobisDistance(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S) {
    // TODO: Implement (z-z_pred)^T * S^-1 * (z-z_pred)
    return 0.0;
}

bool isGated(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S, double threshold) {
    // TODO: Check if dist <= threshold
    return false;
}
