#include "ukf.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

UKF::UKF() {}

void UKF::init(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0, const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in) {
    x = x0;
    P = P0;
    Q = Q_in;
    R = R_in;

    n_x = x.size();
    n_aug = 2 * n_x + 1;
    lambda = 3.0 - n_x;

    weights = Eigen::VectorXd(n_aug);
    weights(0) = lambda / (lambda + n_x);
    for (int i = 1; i < n_aug; ++i) {
        weights(i) = 0.5 / (lambda + n_x);
    }
}

double UKF::normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void UKF::predict(const Eigen::VectorXd& u, double dt) {
    // 1. Generate Sigma Points
    Eigen::MatrixXd Xsig = Eigen::MatrixXd(n_x, n_aug);
    
    // Mean
    Xsig.col(0) = x;
    
    // Square Root of P
    // Using LLT (Cholesky)
    Eigen::MatrixXd L = P.llt().matrixL();
    
    for (int i = 0; i < n_x; ++i) {
        Xsig.col(i + 1)       = x + sqrt(lambda + n_x) * L.col(i);
        Xsig.col(i + 1 + n_x) = x - sqrt(lambda + n_x) * L.col(i);
    }

    // 2. Predict Sigma Points (Process Model)
    // Model: CTRV
    // x' = x + v cos(theta) dt
    // y' = y + v sin(theta) dt
    // theta' = theta + w dt
    
    Xsig_pred = Eigen::MatrixXd(n_x, n_aug);
    double v = u(0);
    double w = u(1);

    for (int i = 0; i < n_aug; ++i) {
        double px = Xsig(0, i);
        double py = Xsig(1, i);
        double theta = Xsig(2, i);

        // Avoid division by zero if using complex models, but simple CTRV is fine
        if (fabs(w) > 0.001) {
            // Complex integral if w != 0
            // px += (v/w) * (sin(theta + w*dt) - sin(theta));
            // py += (v/w) * (cos(theta) - cos(theta + w*dt));
            
            // But let's stick to the simple Euler integration used in Ex04 for consistency
            px += v * cos(theta) * dt;
            py += v * sin(theta) * dt;
            theta += w * dt;
        } else {
            px += v * cos(theta) * dt;
            py += v * sin(theta) * dt;
            theta += w * dt;
        }

        Xsig_pred(0, i) = px;
        Xsig_pred(1, i) = py;
        Xsig_pred(2, i) = normalize_angle(theta);
    }

    // 3. Predict Mean and Covariance
    x.fill(0.0);
    for (int i = 0; i < n_aug; ++i) {
        x += weights(i) * Xsig_pred.col(i);
    }
    // Normalize angle in mean? Usually x(2) needs care.
    // Vector sum of angles is tricky. Let's assume standard weighted sum works if angles are close.
    // Better: atan2(sum sin, sum cos). For now, standard sum then normalize.
    x(2) = normalize_angle(x(2));

    P.fill(0.0);
    for (int i = 0; i < n_aug; ++i) {
        Eigen::VectorXd diff = Xsig_pred.col(i) - x;
        diff(2) = normalize_angle(diff(2));
        P += weights(i) * diff * diff.transpose();
    }
    P += Q;
}

void UKF::update(const Eigen::VectorXd& z) {
    int n_z = z.size();
    
    // 1. Transform Sigma Points to Measurement Space
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, n_aug);
    
    for (int i = 0; i < n_aug; ++i) {
        double px = Xsig_pred(0, i);
        double py = Xsig_pred(1, i);
        double theta = Xsig_pred(2, i);

        // Range
        double r = sqrt(px*px + py*py);
        if (r < 1e-5) r = 1e-5;
        
        // Bearing
        double phi = atan2(py, px) - theta;
        phi = normalize_angle(phi);

        Zsig(0, i) = r;
        Zsig(1, i) = phi;
    }

    // 2. Mean Predicted Measurement
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(n_z);
    for (int i = 0; i < n_aug; ++i) {
        z_pred += weights(i) * Zsig.col(i);
    }
    z_pred(1) = normalize_angle(z_pred(1));

    // 3. Measurement Covariance S
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < n_aug; ++i) {
        Eigen::VectorXd diff = Zsig.col(i) - z_pred;
        diff(1) = normalize_angle(diff(1));
        S += weights(i) * diff * diff.transpose();
    }
    S += R;

    // 4. Cross Covariance T
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(n_x, n_z);
    for (int i = 0; i < n_aug; ++i) {
        Eigen::VectorXd x_diff = Xsig_pred.col(i) - x;
        x_diff(2) = normalize_angle(x_diff(2));

        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalize_angle(z_diff(1));

        Tc += weights(i) * x_diff * z_diff.transpose();
    }

    // 5. Update
    Eigen::MatrixXd K = Tc * S.inverse();
    
    Eigen::VectorXd y = z - z_pred;
    y(1) = normalize_angle(y(1));

    x = x + K * y;
    x(2) = normalize_angle(x(2));
    
    P = P - K * S * K.transpose();
}

Eigen::VectorXd UKF::getState() const {
    return x;
}

Eigen::MatrixXd UKF::getCovariance() const {
    return P;
}
