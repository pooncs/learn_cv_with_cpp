#include <iostream>
#include <opencv2/opencv.hpp>

// TODO: Implement Mouse Callback for selection

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // TODO: Setup Window and Mouse Callback

    while (true) {
        // TODO: Read frame

        // TODO: Convert to HSV

        // TODO: IF Tracking:
        //      1. Calculate Hue Histogram (if first time)
        //      2. Calculate Back Projection
        //      3. Run CamShift/MeanShift
        //      4. Draw Result

        // TODO: Handle user selection drawing (visual feedback)

        if (cv::waitKey(10) == 27) break;
    }

    return 0;
}
