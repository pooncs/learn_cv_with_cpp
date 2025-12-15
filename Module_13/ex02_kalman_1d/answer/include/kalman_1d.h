#pragma once

class KalmanFilter1D {
public:
    KalmanFilter1D(double initial_x, double initial_P, double Q, double R);

    void predict();
    void update(double measurement);

    double getState() const;
    double getCovariance() const;

private:
    double x; // State estimate
    double P; // Estimation error covariance
    double Q; // Process noise covariance
    double R; // Measurement noise covariance
};
