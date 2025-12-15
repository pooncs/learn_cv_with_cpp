#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "kalman_1d.h"

int main() {
    // 1. Setup Simulation
    double true_voltage = 1.25;
    double measurement_noise_std = 0.1;
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, measurement_noise_std);

    // 2. Setup KF
    // Initial guess 0.0, High uncertainty 1.0
    // Process noise very small (we expect constant voltage)
    // Measurement noise matches simulation (0.1^2 = 0.01)
    KalmanFilter1D kf(0.0, 1.0, 1e-5, 0.01);

    std::cout << "Step | True | Meas | Est | P" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    for (int i = 0; i < 20; ++i) {
        // Simulate Measurement
        double z = true_voltage + distribution(generator);

        // KF Cycle
        kf.predict();
        kf.update(z);

        std::cout << std::fixed << std::setprecision(4) 
                  << std::setw(4) << i << " | "
                  << true_voltage << " | "
                  << z << " | "
                  << kf.getState() << " | "
                  << kf.getCovariance() << std::endl;
    }

    // Verification
    if (std::abs(kf.getState() - true_voltage) < 0.1) {
        std::cout << "\nResult: PASS (Converged to " << kf.getState() << ")" << std::endl;
    } else {
        std::cout << "\nResult: FAIL (Did not converge)" << std::endl;
    }

    return 0;
}
