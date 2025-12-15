#include "pipeline.hpp"

namespace cv_curriculum {

void Pipeline::producer() {
    // TODO: Loop while running
    // 1. Create Frame
    // 2. Sleep 33ms
    // 3. Push to queue
}

void Pipeline::consumer() {
    // TODO: Loop while queue.pop(f)
    // 1. Process frame (sleep)
    // 2. Log
}

void Pipeline::start() {
    // TODO: Set running=true
    // Launch threads
}

void Pipeline::stop() {
    // TODO: Set running=false
    // Join threads
}

} // namespace cv_curriculum
