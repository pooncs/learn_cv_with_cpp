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
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(std::move(item));
        }
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
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

} // namespace cv_curriculum
