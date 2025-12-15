#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter();
    
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
    Eigen::VectorXd x; 
    Eigen::MatrixXd P; 
    Eigen::MatrixXd A; 
    Eigen::MatrixXd H; 
    Eigen::MatrixXd Q; 
    Eigen::MatrixXd R; 
    Eigen::MatrixXd I; 
};
