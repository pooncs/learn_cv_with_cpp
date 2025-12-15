#include "buffer_pool.hpp"

namespace cv_curriculum {

// --- BufferHandle ---

BufferHandle::BufferHandle(cv::Mat mat, BufferPool* p) : buffer(mat), pool(p) {}

BufferHandle::~BufferHandle() {
    if (pool) {
        pool->release(buffer);
    }
}

BufferHandle::BufferHandle(BufferHandle&& other) noexcept 
    : buffer(other.buffer), pool(other.pool) {
    other.pool = nullptr; // Take ownership
}

BufferHandle& BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this != &other) {
        if (pool) pool->release(buffer); // Release current
        buffer = other.buffer;
        pool = other.pool;
        other.pool = nullptr;
    }
    return *this;
}

// --- BufferPool ---

BufferPool::BufferPool(int numBuffers, int rows, int cols, int type) {
    for (int i = 0; i < numBuffers; ++i) {
        buffers.push(cv::Mat::zeros(rows, cols, type));
    }
}

BufferHandle BufferPool::acquire() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]{ return !buffers.empty(); });
    
    cv::Mat buf = buffers.top();
    buffers.pop();
    
    return BufferHandle(buf, this);
}

void BufferPool::release(cv::Mat buffer) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        buffers.push(buffer);
    }
    cv.notify_one();
}

size_t BufferPool::available() const {
    // Note: not strictly thread safe if lock not held, but useful for debug
    // We can't make `const` method lock a non-mutable mutex easily without `mutable`
    // Assuming unsafe peek for now or casting away const
    return buffers.size();
}

} // namespace cv_curriculum
