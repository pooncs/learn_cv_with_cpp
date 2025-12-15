#include <gtest/gtest.h>
#include <thread>
#include "resources.hpp"

TEST(DeadlockTest, SafeSwapDoesNotHang) {
    cv_curriculum::Resource r1, r2;
    
    // Launch two threads trying to swap inversely
    // If implementation is correct, they will finish.
    // If not (and we used unsafeSwap), it would hang (test timeout).
    
    std::thread t1(cv_curriculum::safeSwap, std::ref(r1), std::ref(r2));
    std::thread t2(cv_curriculum::safeSwap, std::ref(r2), std::ref(r1));
    
    t1.join();
    t2.join();
    
    SUCCEED();
}
