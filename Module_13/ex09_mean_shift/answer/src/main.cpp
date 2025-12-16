#include <iostream>
#include <opencv2/opencv.hpp>

// Global variables for mouse callback
bool selectObject = false;
int trackObject = 0;
cv::Rect selection;
cv::Point origin;

void onMouse(int event, int x, int y, int, void*) {
    if (selectObject) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= cv::Rect(0, 0, 640, 480); // Clip to image size (assuming 640x480 for simplicity)
    }

    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        origin = cv::Point(x, y);
        selection = cv::Rect(x, y, 0, 0);
        selectObject = true;
        break;
    case cv::EVENT_LBUTTONUP:
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            trackObject = -1; // Ready to start tracking
        break;
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0, 180};
    const float* phranges = hranges;
    cv::Mat frame, hsv, hue, mask, hist, histimg = cv::Mat::zeros(200, 320, CV_8UC3), backproj;
    
    cv::namedWindow("Mean Shift Tracking");
    cv::setMouseCallback("Mean Shift Tracking", onMouse, 0);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Resize for consistency with mouse clip
        cv::resize(frame, frame, cv::Size(640, 480));

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        if (trackObject) {
            int _vmin = 10, _vmax = 256, _smin = 30;

            // Filter out low saturation/value pixels
            cv::inRange(hsv, cv::Scalar(0, _smin, MIN(_vmin, _vmax)),
                        cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);

            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

            if (trackObject < 0) {
                // Initialize Histogram
                cv::Mat roi(hue, selection);
                cv::Mat maskroi(mask, selection);
                cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

                trackWindow = selection;
                trackObject = 1; // Tracking mode on
            }

            // Calculate Back Projection
            cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            backproj &= mask;

            // Mean Shift
            cv::RotatedRect trackBox = cv::CamShift(backproj, trackWindow,
                                    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
            
            // Draw
            if (trackWindow.area() <= 1) {
                int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                trackWindow = cv::Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                              cv::Rect(0, 0, cols, rows);
            }
            cv::ellipse(frame, trackBox, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }

        if (selectObject && selection.width > 0 && selection.height > 0) {
            cv::Mat roi(frame, selection);
            cv::bitwise_not(roi, roi);
        }

        cv::imshow("Mean Shift Tracking", frame);

        char c = (char)cv::waitKey(10);
        if (c == 27) break;
    }

    return 0;
}
