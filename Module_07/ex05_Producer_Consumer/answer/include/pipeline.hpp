#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>

namespace cv_curriculum {

struct Frame {
    int id;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<uint8_t> data; // Dummy data
};

// Reusing SafeQueue logic (inline for simplicity or include from ex04 if linked, but copy is safer for independence)
// We will assume a SafeQueue header exists or paste it here.
// For the sake of the exercise structure, I'll use a minimal thread-safe queue interface here.

template <typename T>
class SafeQueue {
    // ... implementation same as Ex04 ...
    // Since we can't easily include across modules without CMake magic, I'll provide a header-only implementation
    // or assume the student copies it.
    // I will put the full implementation here for the answer.
    #include <queue>
    #include <mutex>
    #include <condition_variable>

    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopped = false;
    size_t maxSize = 0; // 0 = infinite

public:
    SafeQueue(size_t maxS = 0) : maxSize(maxS) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        // Optional: Block if full (backpressure)
        // if (maxSize > 0) cv.wait(lock, [this]{ return queue.size() < maxSize || stopped; });
        queue.push(std::move(item));
        lock.unlock();
        cv.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !queue.empty() || stopped; });
        if (queue.empty() && stopped) return false;
        item = std::move(queue.front());
        queue.pop();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopped = true;
        }
        cv.notify_all();
    }
};

class Pipeline {
    SafeQueue<Frame> queue;
    bool running = false;
    std::vector<std::thread> threads;

    void producer();
    void consumer();

public:
    void start();
    void stop();
};

} // namespace cv_curriculum
