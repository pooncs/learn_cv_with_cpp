#include <gtest/gtest.h>
#include <thread>
#include "counters.hpp"

TEST(RaceTest, SafeCounter) {
    cv_curriculum::SafeCounter c;
    int n = 100;
    int iter = 1000;
    std::vector<std::thread> threads;
    for(int i=0; i<n; ++i) {
        threads.emplace_back([&c, iter](){
            for(int j=0; j<iter; ++j) c.increment();
        });
    }
    for(auto& t : threads) t.join();
    
    EXPECT_EQ(c.get(), n * iter);
}
