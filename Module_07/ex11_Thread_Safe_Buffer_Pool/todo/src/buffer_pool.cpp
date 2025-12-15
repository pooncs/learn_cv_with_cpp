#include "buffer_pool.hpp"

namespace cv_curriculum {

// TODO: Implement BufferHandle methods

BufferPool::BufferPool(int numBuffers, int rows, int cols, int type) {
    // TODO: Pre-allocate buffers
}

BufferHandle BufferPool::acquire() {
    // TODO: Wait for buffer, pop, return handle
    return BufferHandle();
}

void BufferPool::release(cv::Mat buffer) {
    // TODO: Push buffer, notify
}

} // namespace cv_curriculum
