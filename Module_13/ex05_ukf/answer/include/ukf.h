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

    int n_x;
    int n_aug; // 2 * n_x + 1
    double lambda;
    Eigen::VectorXd weights;

    Eigen::MatrixXd Xsig_pred; // Predicted Sigma Points

    // Helper to normalize angle
    double normalize_angle(double angle);
};
