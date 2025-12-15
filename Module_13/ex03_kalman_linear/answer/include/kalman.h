#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter();
    
    // Initialize the filter with matrices and initial state
    void init(const Eigen::VectorXd& x0, 
              const Eigen::MatrixXd& P0, 
              const Eigen::MatrixXd& A, 
              const Eigen::MatrixXd& H, 
              const Eigen::MatrixXd& Q, 
              const Eigen::MatrixXd& R);

    void predict();
    void update(const Eigen::VectorXd& z);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

private:
    Eigen::VectorXd x; // State vector
    Eigen::MatrixXd P; // State covariance
    Eigen::MatrixXd A; // Transition matrix
    Eigen::MatrixXd H; // Measurement matrix
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd R; // Measurement noise covariance
    Eigen::MatrixXd I; // Identity matrix
};
