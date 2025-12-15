#include <gtest/gtest.h>
#include "kalman_1d.h"

TEST(KalmanTest, PredictIncreasesUncertainty) {
    double initial_P = 1.0;
    double Q = 0.1;
    KalmanFilter1D kf(0.0, initial_P, Q, 0.1);
    
    kf.predict();
    
    EXPECT_GT(kf.getCovariance(), initial_P);
    EXPECT_NEAR(kf.getCovariance(), initial_P + Q, 1e-5);
}

TEST(KalmanTest, UpdateDecreasesUncertainty) {
    double initial_P = 1.0;
    double R = 0.5;
    KalmanFilter1D kf(0.0, initial_P, 0.0, R); // Q=0 to isolate update
    
    kf.update(1.0);
    
    EXPECT_LT(kf.getCovariance(), initial_P);
}

TEST(KalmanTest, ConvergesToValue) {
    KalmanFilter1D kf(0.0, 1.0, 1e-5, 0.1);
    double val = 5.0;
    
    for(int i=0; i<50; ++i) {
        kf.predict();
        kf.update(val);
    }
    
    EXPECT_NEAR(kf.getState(), val, 0.1);
}
