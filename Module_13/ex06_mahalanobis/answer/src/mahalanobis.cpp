#include "mahalanobis.h"

double computeMahalanobisDistance(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S) {
    Eigen::VectorXd diff = z - z_pred;
    // d^2 = y^T S^-1 y
    return diff.transpose() * S.inverse() * diff;
}

bool isGated(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S, double threshold) {
    double d2 = computeMahalanobisDistance(z, z_pred, S);
    return d2 <= threshold;
}
