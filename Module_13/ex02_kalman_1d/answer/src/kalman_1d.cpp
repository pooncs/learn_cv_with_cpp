#include "kalman_1d.h"

KalmanFilter1D::KalmanFilter1D(double initial_x, double initial_P, double Q, double R)
    : x(initial_x), P(initial_P), Q(Q), R(R) {}

void KalmanFilter1D::predict() {
    // x_k = x_{k-1} (Constant model)
    // P_k = P_{k-1} + Q
    P = P + Q;
}

void KalmanFilter1D::update(double z) {
    // K = P / (P + R)
    double K = P / (P + R);
    
    // x = x + K(z - x)
    x = x + K * (z - x);
    
    // P = (1 - K)P
    P = (1.0 - K) * P;
}

double KalmanFilter1D::getState() const {
    return x;
}

double KalmanFilter1D::getCovariance() const {
    return P;
}
