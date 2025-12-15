#pragma once
#include <mutex>
#include <thread>
#include <chrono>

namespace cv_curriculum {

struct Resource {
    std::mutex mtx;
    int data = 0;
};

// Deadlock-prone function
void unsafeSwap(Resource& r1, Resource& r2);

// Safe swap function
void safeSwap(Resource& r1, Resource& r2);

} // namespace cv_curriculum
