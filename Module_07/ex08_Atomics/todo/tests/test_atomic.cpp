#include <gtest/gtest.h>
#include <thread>
#include "atomic_counter.hpp"

TEST(AtomicTest, IsAtomic) {
    cv_curriculum::AtomicCounter counter;
    int n = 100;
    int iter = 1000;
    std::vector<std::thread> threads;
    for(int i=0; i<n; ++i) {
        threads.emplace_back([&counter, iter](){
            for(int j=0; j<iter; ++j) counter.increment();
        });
    }
    for(auto& t : threads) t.join();
    
    // If it's not atomic, this will likely fail
    EXPECT_EQ(counter.get(), n * iter);
}
