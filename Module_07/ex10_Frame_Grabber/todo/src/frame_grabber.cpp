#include "frame_grabber.hpp"
#include <chrono>

namespace cv_curriculum {

FrameGrabber::FrameGrabber() {}

FrameGrabber::~FrameGrabber() {
    stop();
}

void FrameGrabber::start() {
    // TODO: Launch thread
}

void FrameGrabber::stop() {
    // TODO: Stop thread
}

bool FrameGrabber::getLatest(cv::Mat& out) {
    // TODO: Lock and copy
    return false;
}

void FrameGrabber::grabLoop() {
    // TODO: Loop while running
    // Simulate grab
    // Update latestFrame with lock
}

} // namespace cv_curriculum
