#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "kalman.h"

TEST(LinearKFTest, StaticPrediction) {
    KalmanFilter kf;
    
    // Simple 1D case
    Eigen::VectorXd x(1); x << 10.0;
    Eigen::MatrixXd P(1,1); P << 1.0;
    Eigen::MatrixXd A(1,1); A << 1.0;
    Eigen::MatrixXd H(1,1); H << 1.0;
    Eigen::MatrixXd Q(1,1); Q << 0.1;
    Eigen::MatrixXd R(1,1); R << 0.1;
    
    kf.init(x, P, A, H, Q, R);
    
    kf.predict();
    
    // x should be same (A=1)
    EXPECT_NEAR(kf.getState()(0), 10.0, 1e-5);
    // P should increase (P+Q)
    EXPECT_NEAR(kf.getCovariance()(0,0), 1.1, 1e-5);
}

TEST(LinearKFTest, SimpleUpdate) {
    KalmanFilter kf;
    Eigen::VectorXd x(1); x << 0.0;
    Eigen::MatrixXd P(1,1); P << 1.0;
    Eigen::MatrixXd A(1,1); A << 1.0;
    Eigen::MatrixXd H(1,1); H << 1.0;
    Eigen::MatrixXd Q(1,1); Q << 0.0;
    Eigen::MatrixXd R(1,1); R << 1.0;
    
    kf.init(x, P, A, H, Q, R);
    
    Eigen::VectorXd z(1); z << 2.0;
    kf.update(z);
    
    // K = 1 / (1+1) = 0.5
    // x = 0 + 0.5(2-0) = 1.0
    EXPECT_NEAR(kf.getState()(0), 1.0, 1e-5);
}
