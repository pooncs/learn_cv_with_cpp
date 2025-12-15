#pragma once
#include <Eigen/Dense>

double computeMahalanobisDistance(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S);
bool isGated(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S, double threshold);
