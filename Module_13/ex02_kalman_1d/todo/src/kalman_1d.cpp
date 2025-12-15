#include "kalman_1d.h"

KalmanFilter1D::KalmanFilter1D(double initial_x, double initial_P, double Q, double R)
    : x(initial_x), P(initial_P), Q(Q), R(R) {}

void KalmanFilter1D::predict() {
    // TODO: Implement Predict Step
    // P = P + Q
}

void KalmanFilter1D::update(double z) {
    // TODO: Implement Update Step
    // K = ...
    // x = ...
    // P = ...
}

double KalmanFilter1D::getState() const {
    return x;
}

double KalmanFilter1D::getCovariance() const {
    return P;
}
