#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "kalman.h"

int main() {
    // 1. Setup Constants
    double dt = 0.1;
    double meas_noise = 0.5; // Position noise
    double accel_noise = 0.1; // Process noise magnitude

    // 2. Setup Matrices (CV Model: x, y, vx, vy)
    int n = 4; // State dim
    int m = 2; // Meas dim (x, y)

    Eigen::MatrixXd A(n, n);
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;

    Eigen::MatrixXd H(m, n);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * accel_noise;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * meas_noise;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 1.0;
    
    Eigen::VectorXd x0(n);
    x0 << 0, 0, 0, 0; // Initial guess

    KalmanFilter kf;
    kf.init(x0, P0, A, H, Q, R);

    // 3. Simulation
    std::default_random_engine gen;
    std::normal_distribution<double> noise(0.0, sqrt(meas_noise));

    double true_vx = 1.0;
    double true_vy = 0.5;
    double true_x = 0.0;
    double true_y = 0.0;

    std::cout << "Time | True(x,y) | Meas(x,y) | Est(x,y) | Est(vx,vy)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int t = 0; t < 20; ++t) {
        // Update Truth
        true_x += true_vx * dt;
        true_y += true_vy * dt;

        // Simulate Measurement
        Eigen::VectorXd z(m);
        z << true_x + noise(gen), 
             true_y + noise(gen);

        // KF Cycle
        kf.predict();
        kf.update(z);

        Eigen::VectorXd est = kf.getState();

        std::cout << std::fixed << std::setprecision(2)
                  << t*dt << " | "
                  << "(" << true_x << "," << true_y << ") | "
                  << "(" << z(0) << "," << z(1) << ") | "
                  << "(" << est(0) << "," << est(1) << ") | "
                  << "(" << est(2) << "," << est(3) << ")" << std::endl;
    }

    // Verification
    Eigen::VectorXd est = kf.getState();
    double vel_err = sqrt(pow(est(2) - true_vx, 2) + pow(est(3) - true_vy, 2));
    
    std::cout << "\nVelocity Error: " << vel_err << std::endl;

    if (vel_err < 0.5) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL (Velocity did not converge)" << std::endl;
    }

    return 0;
}
