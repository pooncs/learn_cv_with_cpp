#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "probability.h"

TEST(ProbabilityTest, ConvolutionSumIsConserved) {
    std::vector<double> belief(10, 0.0);
    belief[5] = 1.0;
    std::vector<double> kernel = {0.1, 0.8, 0.1};
    
    std::vector<double> result = convolve1D(belief, kernel);
    
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST(ProbabilityTest, ShiftsPeakCorrectly) {
    std::vector<double> belief(10, 0.0);
    belief[5] = 1.0;
    // Kernel that shifts right by 1: [0, 0, 1] means kernel[2] (offset=1) is 1.0
    // Actually our code assumes center at K/2.
    // If K=3, center index is 1.
    // {0, 0, 1} means index 2 has prob 1. Index 2 is offset + 1.
    // So this should move +1.
    std::vector<double> kernel = {0.0, 0.0, 1.0}; 
    
    std::vector<double> result = convolve1D(belief, kernel);
    
    EXPECT_NEAR(result[6], 1.0, 1e-5);
    EXPECT_NEAR(result[5], 0.0, 1e-5);
}
