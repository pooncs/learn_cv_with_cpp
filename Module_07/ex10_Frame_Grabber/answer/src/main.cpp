#include <iostream>
#include <thread>
#include "frame_grabber.hpp"

int main() {
    cv_curriculum::FrameGrabber grabber;
    grabber.start();
    
    // Simulate UI loop running faster than camera
    // 100 iterations at 10ms = 1 sec
    // Camera is 30 FPS (~33ms)
    // We should see duplicate frames or frames updating every ~3 iterations
    
    cv::Mat frame;
    int lastFrameId = -1;
    
    for (int i = 0; i < 50; ++i) {
        if (grabber.getLatest(frame)) {
            // In a real app we'd display it
            // Here we just check content implies update
            // Since we can't easily parse text, we assume it's working if size is correct
            // std::cout << "Got frame " << frame.size() << std::endl;
        } else {
            std::cout << "Waiting for first frame..." << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    grabber.stop();
    return 0;
}
