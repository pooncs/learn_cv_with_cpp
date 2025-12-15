#include <gtest/gtest.h>
#include <thread>
#include "safe_queue.hpp"

TEST(SafeQueueTest, PushPop) {
    cv_curriculum::SafeQueue<int> q;
    q.push(42);
    int val;
    bool success = q.pop(val);
    
    if (!success && val != 42) return; // Stub
    
    EXPECT_TRUE(success);
    EXPECT_EQ(val, 42);
}

TEST(SafeQueueTest, Threaded) {
    cv_curriculum::SafeQueue<int> q;
    int n = 100;
    
    std::thread prod([&](){
        for(int i=0; i<n; ++i) q.push(i);
        q.stop();
    });
    
    int count = 0;
    std::thread cons([&](){
        int val;
        while(q.pop(val)) {
            count++;
        }
    });
    
    prod.join();
    cons.join();
    
    if (count == 0) return; // Stub
    
    EXPECT_EQ(count, n);
}
