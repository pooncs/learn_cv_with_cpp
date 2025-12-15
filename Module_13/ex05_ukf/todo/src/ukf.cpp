#include "ukf.h"
#include <iostream>

UKF::UKF() {}

void UKF::init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in) {
    // TODO: Init parameters
}

void UKF::predict(const Eigen::VectorXd& u, double dt) {
    // TODO: Generate Sigma Points
    
    // TODO: Predict Sigma Points
    
    // TODO: Predict Mean and Covariance
}

void UKF::update(const Eigen::VectorXd& z) {
    // TODO: Transform Sigma Points to Meas Space
    
    // TODO: Calc Mean and Covariance
    
    // TODO: Update
}

Eigen::VectorXd UKF::getState() const {
    return x;
}

Eigen::MatrixXd UKF::getCovariance() const {
    return P;
}
