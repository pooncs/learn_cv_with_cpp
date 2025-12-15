#pragma once
#include <Eigen/Dense>

class EKF {
public:
    EKF();
    
    // Init state [x, y, theta]
    void init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

    // Predict using control input u = [v, omega]
    void predict(const Eigen::VectorXd& u, double dt);

    // Update using measurement z = [range, bearing]
    void update(const Eigen::VectorXd& z);

    Eigen::VectorXd getState() const;
    Eigen::MatrixXd getCovariance() const;

private:
    Eigen::VectorXd x; // State
    Eigen::MatrixXd P; // Covariance
    Eigen::MatrixXd Q; // Process Noise
    Eigen::MatrixXd R; // Measurement Noise
};
