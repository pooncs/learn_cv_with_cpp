#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

namespace cv_curriculum {

template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopped = false;

public:
    void push(T item) {
        // TODO: Lock, push, notify
    }

    bool pop(T& item) {
        // TODO: Lock, wait, pop
        // If stopped and empty, return false
        return false;
    }

    void stop() {
        // TODO: Set stopped flag and notify all
    }
};

} // namespace cv_curriculum
