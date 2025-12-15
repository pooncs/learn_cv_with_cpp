#include "kalman.h"

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(const Eigen::VectorXd& x0, 
                        const Eigen::MatrixXd& P0, 
                        const Eigen::MatrixXd& A_in, 
                        const Eigen::MatrixXd& H_in, 
                        const Eigen::MatrixXd& Q_in, 
                        const Eigen::MatrixXd& R_in) 
{
    // TODO: Initialize matrices
    // x = x0;
    // ...
    // I = Eigen::MatrixXd::Identity(x.size(), x.size());
}

void KalmanFilter::predict() {
    // TODO: Predict
    // x = A * x
    // P = ...
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    // TODO: Update
    // y = z - Hx
    // S = HPH' + R
    // K = PH'S^-1
    // x = x + Ky
    // P = (I - KH)P
}

Eigen::VectorXd KalmanFilter::getState() const {
    return x;
}

Eigen::MatrixXd KalmanFilter::getCovariance() const {
    return P;
}
