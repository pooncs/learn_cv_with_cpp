#include <iostream>
#include <vector>
#include <iomanip>
#include "mahalanobis.h"

int main() {
    // Setup
    Eigen::VectorXd z_pred(2); z_pred << 0.0, 0.0;
    
    // Covariance: High uncertainty in X (variance=4), Low in Y (variance=0.25)
    Eigen::MatrixXd S(2, 2);
    S << 4.0, 0.0,
         0.0, 0.25;

    // Chi-Square threshold for 2 DOF, 95% confidence is ~5.99
    double threshold = 5.99;

    std::vector<Eigen::VectorXd> measurements;
    // 1. Point at (3, 0) -> Euclidean=3. Mahalanobis: 3^2 / 4 = 2.25. (Accept)
    Eigen::VectorXd p1(2); p1 << 3.0, 0.0; measurements.push_back(p1);

    // 2. Point at (0, 1.5) -> Euclidean=1.5. Mahalanobis: 1.5^2 / 0.25 = 2.25 / 0.25 = 9.0. (Reject)
    Eigen::VectorXd p2(2); p2 << 0.0, 1.5; measurements.push_back(p2);

    // 3. Point at (4, 0) -> Mahalanobis: 16/4 = 4.0 (Accept)
    Eigen::VectorXd p3(2); p3 << 4.0, 0.0; measurements.push_back(p3);

    // 4. Point at (5, 0) -> Mahalanobis: 25/4 = 6.25 (Reject)
    Eigen::VectorXd p4(2); p4 << 5.0, 0.0; measurements.push_back(p4);

    std::cout << "Pred: (0,0), Cov: diag(4, 0.25)" << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    std::cout << "Point | Euclid | Mahal2 | Gated?" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    bool all_correct = true;

    for (const auto& z : measurements) {
        double d_euc = z.norm(); // dist from 0
        double d_mah2 = computeMahalanobisDistance(z, z_pred, S);
        bool gated = isGated(z, z_pred, S, threshold);
        
        std::cout << "(" << z(0) << "," << z(1) << ") | "
                  << std::fixed << std::setprecision(2) << d_euc << " | "
                  << d_mah2 << " | "
                  << (gated ? "YES" : "NO") << std::endl;

        // Verification logic
        if (z(0) == 3.0 && !gated) all_correct = false;
        if (z(1) == 1.5 && gated) all_correct = false;
        if (z(0) == 4.0 && !gated) all_correct = false;
        if (z(0) == 5.0 && gated) all_correct = false;
    }

    if (all_correct) {
        std::cout << "\nResult: PASS" << std::endl;
    } else {
        std::cout << "\nResult: FAIL" << std::endl;
    }

    return 0;
}
