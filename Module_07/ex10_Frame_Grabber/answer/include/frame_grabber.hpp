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
    
    /**
     * @brief Gets the latest frame.
     * @param out Output frame.
     * @return true if a frame is available, false otherwise.
     */
    bool getLatest(cv::Mat& out);

private:
    void grabLoop();

    std::thread grabThread;
    std::mutex mtx;
    cv::Mat latestFrame;
    std::atomic<bool> running{false};
    std::atomic<bool> hasNewFrame{false};
};

} // namespace cv_curriculum
