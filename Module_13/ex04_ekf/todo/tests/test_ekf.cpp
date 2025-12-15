#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "ekf.h"

TEST(EKFTest, StationaryUpdate) {
    EKF ekf;
    Eigen::VectorXd x0(3); x0 << 1.0, 0.0, 0.0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(3,3) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2,2) * 0.01;
    
    ekf.init(x0, P0, Q, R);
    
    // Measure exactly what we expect: r=1, bearing=0 (landmark at 0,0, robot at 1,0 facing 0)
    // Wait... landmark at (0,0). Robot at (1,0).
    // Range = 1.
    // Bearing = atan2(0, 1) - 0 = 0 - 0 = 0. 
    // BUT bearing is relative to robot heading. atan2(y,x) gives angle of vector from robot to landmark? 
    // No, standard is usually atan2(dy, dx). 
    // Here landmark is (0,0). Robot is (px, py).
    // Vector robot->landmark is (-px, -py).
    // atan2(0-0, 0-1) = atan2(0, -1) = pi.
    // So expected bearing is PI.
    
    Eigen::VectorXd z(2); z << 1.0, 3.14159;
    
    ekf.update(z);
    
    // Should stay roughly same
    EXPECT_NEAR(ekf.getState()(0), 1.0, 0.1);
}
