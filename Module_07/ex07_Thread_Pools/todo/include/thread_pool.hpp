#pragma once
#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace cv_curriculum {

class ThreadPool {
public:
    using Task = std::function<void()>;

    ThreadPool(size_t numThreads);
    ~ThreadPool();

    void enqueue(Task task);

private:
    std::vector<std::thread> workers;
    // TODO: Add queue, mutex, condition variable, stop flag
};

} // namespace cv_curriculum
