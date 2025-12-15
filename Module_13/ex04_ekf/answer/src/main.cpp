#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include "ekf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    double dt = 0.1;
    
    // Init EKF
    Eigen::VectorXd x0(3); x0 << 5.0, 0.0, M_PI/2.0; // Start at (5,0) facing North
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(3, 3) * 0.1;
    
    // Process Noise (Uncertainty in v, w mapped to x, y, theta)
    // Simplified Q for state
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(3, 3) * 0.01;
    
    // Measurement Noise (Range, Bearing)
    Eigen::MatrixXd R(2, 2);
    R << 0.1, 0.0,
         0.0, 0.01; // 0.1m range noise, 0.01 rad bearing noise

    EKF ekf;
    ekf.init(x0, P0, Q, R);

    // Simulation
    std::default_random_engine gen;
    std::normal_distribution<double> n_r(0.0, sqrt(R(0,0)));
    std::normal_distribution<double> n_b(0.0, sqrt(R(1,1)));

    double v = 1.0;
    double w = 0.1; // Turn rate
    
    // Truth
    double tx = 5.0;
    double ty = 0.0;
    double tt = M_PI/2.0;

    std::cout << "Time | True(x,y) | Est(x,y)" << std::endl;
    std::cout << "---------------------------" << std::endl;

    for (int k = 0; k < 50; ++k) {
        // Move Truth
        tx += v * cos(tt) * dt;
        ty += v * sin(tt) * dt;
        tt += w * dt;

        // Generate Meas
        double tr = sqrt(tx*tx + ty*ty);
        double tb = atan2(ty, tx) - tt;
        while(tb > M_PI) tb -= 2*M_PI;
        while(tb < -M_PI) tb += 2*M_PI;

        Eigen::VectorXd z(2);
        z << tr + n_r(gen), tb + n_b(gen);

        // EKF Predict
        Eigen::VectorXd u(2); u << v, w;
        ekf.predict(u, dt);

        // EKF Update
        ekf.update(z);

        Eigen::VectorXd est = ekf.getState();
        
        std::cout << std::fixed << std::setprecision(2)
                  << k*dt << " | "
                  << "(" << tx << "," << ty << ") | "
                  << "(" << est(0) << "," << est(1) << ")" << std::endl;
    }

    Eigen::VectorXd final_est = ekf.getState();
    double dist_err = sqrt(pow(final_est(0) - tx, 2) + pow(final_est(1) - ty, 2));
    
    std::cout << "\nFinal Error: " << dist_err << std::endl;
    if (dist_err < 0.5) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    return 0;
}
