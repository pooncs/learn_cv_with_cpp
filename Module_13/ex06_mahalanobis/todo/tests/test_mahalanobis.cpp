#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "mahalanobis.h"

TEST(MahalanobisTest, ComputesCorrectly) {
    Eigen::VectorXd z(2); z << 2.0, 0.0;
    Eigen::VectorXd z_pred(2); z_pred << 0.0, 0.0;
    Eigen::MatrixXd S(2, 2); S << 4.0, 0.0, 0.0, 1.0;
    
    // diff = [2, 0]
    // S^-1 = [0.25, 0; 0, 1]
    // diff^T * S^-1 * diff = [2, 0] * [0.5; 0] = 1.0
    
    double d2 = computeMahalanobisDistance(z, z_pred, S);
    EXPECT_NEAR(d2, 1.0, 1e-5);
}

TEST(MahalanobisTest, GatingWorks) {
    Eigen::VectorXd z(2); z << 10.0, 0.0;
    Eigen::VectorXd z_pred(2); z_pred << 0.0, 0.0;
    Eigen::MatrixXd S(2, 2); S << 1.0, 0.0, 0.0, 1.0;
    
    // d2 = 100. Threshold = 5.99. Should Reject.
    EXPECT_FALSE(isGated(z, z_pred, S, 5.99));
    
    Eigen::VectorXd z2(2); z2 << 1.0, 0.0;
    // d2 = 1. Should Accept.
    EXPECT_TRUE(isGated(z2, z_pred, S, 5.99));
}
