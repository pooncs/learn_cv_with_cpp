#pragma once
#include <opencv2/opencv.hpp>
#include <stack>
#include <mutex>
#include <condition_variable>

namespace cv_curriculum {

class BufferPool;

class BufferHandle {
public:
    // TODO: Implement move semantics and destructor that releases buffer
    BufferHandle() = default;
    // ...
private:
    // cv::Mat buffer;
    // BufferPool* pool;
};

class BufferPool {
public:
    BufferPool(int numBuffers, int rows, int cols, int type);
    
    // TODO: Implement acquire and release
    BufferHandle acquire();
    void release(cv::Mat buffer);

private:
    std::stack<cv::Mat> buffers;
    std::mutex mtx;
    std::condition_variable cv;
};

} // namespace cv_curriculum
