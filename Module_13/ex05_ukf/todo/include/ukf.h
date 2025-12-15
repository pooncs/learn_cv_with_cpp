#pragma once
#include <Eigen/Dense>

class UKF {
public:
    UKF();
    
    void init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

    void predict(const Eigen::VectorXd& u, double dt);
    void update(const Eigen::VectorXd& z);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

private:
    Eigen::VectorXd x; 
    Eigen::MatrixXd P; 
    Eigen::MatrixXd Q; 
    Eigen::MatrixXd R;

    // TODO: Add UKF parameters (lambda, weights, etc.)
};
