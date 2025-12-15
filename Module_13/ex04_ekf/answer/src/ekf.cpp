#include "ekf.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to normalize angle to [-pi, pi]
double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

EKF::EKF() {}

void EKF::init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in) {
    x = x0;
    P = P0;
    Q = Q_in;
    R = R_in;
}

void EKF::predict(const Eigen::VectorXd& u, double dt) {
    double v = u(0);
    double w = u(1);
    double theta = x(2);

    // 1. Predict State (Non-linear)
    // x' = x + v cos(theta) dt
    // y' = y + v sin(theta) dt
    // theta' = theta + w dt
    Eigen::VectorXd x_pred(3);
    x_pred(0) = x(0) + v * cos(theta) * dt;
    x_pred(1) = x(1) + v * sin(theta) * dt;
    x_pred(2) = x(2) + w * dt;
    x_pred(2) = normalize_angle(x_pred(2));

    // 2. Jacobian F
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(3, 3);
    F(0, 2) = -v * sin(theta) * dt;
    F(1, 2) = v * cos(theta) * dt;

    // 3. Predict Covariance
    P = F * P * F.transpose() + Q;
    x = x_pred;
}

void EKF::update(const Eigen::VectorXd& z) {
    // Landmark at (0,0)
    double px = x(0);
    double py = x(1);
    double theta = x(2);

    // 1. Predicted Measurement (Non-linear h(x))
    // r = sqrt(x^2 + y^2)
    // phi = atan2(y, x) - theta (bearing relative to robot)
    double range = sqrt(px * px + py * py);
    
    // Avoid division by zero
    if (range < 1e-5) range = 1e-5;

    double bearing = atan2(py, px) - theta;
    bearing = normalize_angle(bearing);

    Eigen::VectorXd z_pred(2);
    z_pred << range, bearing;

    // 2. Innovation
    Eigen::VectorXd y = z - z_pred;
    y(1) = normalize_angle(y(1));

    // 3. Jacobian H
    // H = [ dr/dx      dr/dy      dr/dtheta ]
    //     [ dphi/dx    dphi/dy    dphi/dtheta ]
    
    Eigen::MatrixXd H(2, 3);
    // dr/dx = x / r, dr/dy = y / r, dr/dtheta = 0
    H(0, 0) = px / range;
    H(0, 1) = py / range;
    H(0, 2) = 0.0;

    // dphi/dx = -y/r^2, dphi/dy = x/r^2, dphi/dtheta = -1
    double r2 = range * range;
    H(1, 0) = -py / r2;
    H(1, 1) = px / r2;
    H(1, 2) = -1.0;

    // 4. Update Steps
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    x = x + K * y;
    x(2) = normalize_angle(x(2));
    
    P = (Eigen::MatrixXd::Identity(3, 3) - K * H) * P;
}

Eigen::VectorXd EKF::getState() const {
    return x;
}

Eigen::MatrixXd EKF::getCovariance() const {
    return P;
}
