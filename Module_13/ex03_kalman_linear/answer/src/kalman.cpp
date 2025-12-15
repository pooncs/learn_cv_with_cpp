#include "kalman.h"

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(const Eigen::VectorXd& x0, 
                        const Eigen::MatrixXd& P0, 
                        const Eigen::MatrixXd& A_in, 
                        const Eigen::MatrixXd& H_in, 
                        const Eigen::MatrixXd& Q_in, 
                        const Eigen::MatrixXd& R_in) 
{
    x = x0;
    P = P0;
    A = A_in;
    H = H_in;
    Q = Q_in;
    R = R_in;
    I = Eigen::MatrixXd::Identity(x.size(), x.size());
}

void KalmanFilter::predict() {
    // x = A * x
    x = A * x;
    // P = A * P * A^T + Q
    P = A * P * A.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    // y = z - H * x (Innovation)
    Eigen::VectorXd y = z - H * x;
    
    // S = H * P * H^T + R (Innovation Covariance)
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    
    // K = P * H^T * S^-1 (Kalman Gain)
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    
    // x = x + K * y
    x = x + K * y;
    
    // P = (I - K * H) * P
    P = (I - K * H) * P;
}

Eigen::VectorXd KalmanFilter::getState() const {
    return x;
}

Eigen::MatrixXd KalmanFilter::getCovariance() const {
    return P;
}
