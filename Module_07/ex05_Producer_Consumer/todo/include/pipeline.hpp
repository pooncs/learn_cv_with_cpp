#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace cv_curriculum {

struct Frame {
    int id;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<uint8_t> data;
};

template <typename T>
class SafeQueue {
    // TODO: Copy your SafeQueue implementation from Ex04 or re-implement
    // For Todo, you can use std::queue + mutex + cv stubs
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopped = false;
public:
    void push(T item) {
        // TODO
    }
    bool pop(T& item) {
        // TODO
        return false;
    }
    void stop() {
        // TODO
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
