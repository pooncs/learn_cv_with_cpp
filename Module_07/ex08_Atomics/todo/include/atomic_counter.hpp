#pragma once
#include <atomic>

namespace cv_curriculum {

struct AtomicCounter {
    // TODO: Change int to std::atomic<int>
    int value{0};

    void increment() {
        // TODO: Atomic increment
        value++; 
    }

    int get() const {
        return value;
    }
};

} // namespace cv_curriculum
