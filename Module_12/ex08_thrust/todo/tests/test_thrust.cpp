#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

TEST(ThrustTest, BasicReduce) {
    thrust::host_vector<int> h_vec(100);
    std::fill(h_vec.begin(), h_vec.end(), 1);
    
    thrust::device_vector<int> d_vec = h_vec;
    
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
    
    EXPECT_EQ(sum, 100);
}
