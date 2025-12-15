#include "thread_pool.hpp"

namespace cv_curriculum {

ThreadPool::ThreadPool(size_t numThreads) {
    // TODO: Launch workers
    // Worker loop:
    // 1. Lock
    // 2. Wait until stop or !empty
    // 3. Pop task
    // 4. Unlock
    // 5. Execute task
}

ThreadPool::~ThreadPool() {
    // TODO: Signal stop, notify all, join all
}

void ThreadPool::enqueue(Task task) {
    // TODO: Lock, push, notify
}

} // namespace cv_curriculum
