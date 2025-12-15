#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace cv_curriculum {

class BufferPool;

// RAII Wrapper
class BufferHandle {
public:
    BufferHandle() = default;
    BufferHandle(cv::Mat mat, BufferPool* p);
    ~BufferHandle();
    
    // Move only
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    BufferHandle(BufferHandle&& other) noexcept;
    BufferHandle& operator=(BufferHandle&& other) noexcept;

    cv::Mat& get() { return buffer; }
    bool isValid() const { return pool != nullptr; }

private:
    cv::Mat buffer;
    BufferPool* pool = nullptr;
};

class BufferPool {
public:
    BufferPool(int numBuffers, int rows, int cols, int type);
    
    /**
     * @brief Acquires a buffer. Blocks if empty.
     */
    BufferHandle acquire();
    
    /**
     * @brief Returns a buffer to the pool.
     * Usually called by BufferHandle destructor.
     */
    void release(cv::Mat buffer);

    size_t available() const;

private:
    std::stack<cv::Mat> buffers;
    std::mutex mtx;
    std::condition_variable cv;
};

} // namespace cv_curriculum
