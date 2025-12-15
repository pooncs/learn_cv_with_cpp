#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "ukf.h"

TEST(UKFTest, StationaryUpdate) {
    UKF ukf;
    Eigen::VectorXd x0(3); x0 << 1.0, 0.0, 0.0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(3,3) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2,2) * 0.01;
    
    ukf.init(x0, P0, Q, R);
    
    // Range=1, Bearing=PI (see EKF test)
    Eigen::VectorXd z(2); z << 1.0, 3.14159;
    
    ukf.update(z);
    
    EXPECT_NEAR(ukf.getState()(0), 1.0, 0.1);
}
