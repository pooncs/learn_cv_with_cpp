#include <gtest/gtest.h>
#include <thread>
#include "frame_grabber.hpp"

TEST(GrabberTest, StartsAndStops) {
    cv_curriculum::FrameGrabber grabber;
    grabber.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    grabber.stop();
    SUCCEED();
}
