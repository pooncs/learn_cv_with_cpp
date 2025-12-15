#include "resources.hpp"

namespace cv_curriculum {

void unsafeSwap(Resource& r1, Resource& r2) {
    // This is prone to deadlock if another thread swaps(r2, r1)
    std::lock_guard<std::mutex> lock1(r1.mtx);
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Induce deadlock chance
    std::lock_guard<std::mutex> lock2(r2.mtx);
    
    int temp = r1.data;
    r1.data = r2.data;
    r2.data = temp;
}

void safeSwap(Resource& r1, Resource& r2) {
    // C++17 scoped_lock uses deadlock avoidance algo
    std::scoped_lock lock(r1.mtx, r2.mtx);
    
    int temp = r1.data;
    r1.data = r2.data;
    r2.data = temp;
}

} // namespace cv_curriculum
