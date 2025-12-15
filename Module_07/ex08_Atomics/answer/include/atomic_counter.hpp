#pragma once
#include <atomic>

namespace cv_curriculum {

struct AtomicCounter {
    std::atomic<int> value{0};

    void increment() {
        value++; // Atomic increment
    }

    int get() const {
        return value.load();
    }
};

} // namespace cv_curriculum
