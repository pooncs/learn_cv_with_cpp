#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>

namespace cv_curriculum {

class FrameGrabber {
public:
    FrameGrabber();
    ~FrameGrabber();

    void start();
    void stop();
    
    bool getLatest(cv::Mat& out);

private:
    void grabLoop();

    std::thread grabThread;
    std::mutex mtx;
    cv::Mat latestFrame;
    std::atomic<bool> running{false};
    // TODO: Add flag if needed
};

} // namespace cv_curriculum
