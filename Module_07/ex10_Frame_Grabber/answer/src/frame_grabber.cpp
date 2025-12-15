#include "frame_grabber.hpp"
#include <chrono>

namespace cv_curriculum {

FrameGrabber::FrameGrabber() {}

FrameGrabber::~FrameGrabber() {
    stop();
}

void FrameGrabber::start() {
    if (running) return;
    running = true;
    grabThread = std::thread(&FrameGrabber::grabLoop, this);
}

void FrameGrabber::stop() {
    running = false;
    if (grabThread.joinable()) {
        grabThread.join();
    }
}

bool FrameGrabber::getLatest(cv::Mat& out) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!hasNewFrame) return false;
    
    // Copy the frame
    out = latestFrame.clone(); 
    // In a real optimized scenario, we might swap pointers or use shared_ptr to avoid deep copy
    
    // Optional: Reset flag if we only want to consume once
    // hasNewFrame = false; 
    
    return true;
}

void FrameGrabber::grabLoop() {
    int counter = 0;
    while (running) {
        // Simulate grab (e.g., 30 FPS -> 33ms)
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        
        cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame, "Frame " + std::to_string(counter++), cv::Point(50, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            latestFrame = frame;
            hasNewFrame = true;
        }
    }
}

} // namespace cv_curriculum
