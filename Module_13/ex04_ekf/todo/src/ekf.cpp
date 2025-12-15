#include "ekf.h"
#include <iostream>
#include <cmath>

EKF::EKF() {}

void EKF::init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in) {
    // TODO: Init
}

void EKF::predict(const Eigen::VectorXd& u, double dt) {
    // TODO: Predict State (Non-linear)
    
    // TODO: Jacobian F
    
    // TODO: Predict Covariance
}

void EKF::update(const Eigen::VectorXd& z) {
    // TODO: Predicted Measurement (h(x))
    
    // TODO: Innovation y = z - h(x)
    
    // TODO: Jacobian H
    
    // TODO: Update Steps
}

Eigen::VectorXd EKF::getState() const {
    return x;
}

Eigen::MatrixXd EKF::getCovariance() const {
    return P;
}
