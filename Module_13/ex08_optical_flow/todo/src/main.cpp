#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // TODO: Read first frame and convert to gray

    // TODO: Detect features to track (goodFeaturesToTrack)

    while (true) {
        // TODO: Read next frame

        // TODO: Calculate Optical Flow (calcOpticalFlowPyrLK)

        // TODO: Select good points and draw tracks

        // TODO: Update old_gray and p0

        if (cv::waitKey(30) == 'q') break;
    }
    return 0;
}
