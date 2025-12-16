#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    // Parameters for Shi-Tomasi corner detection
    std::vector<cv::Point2f> p0, p1;
    cv::Mat old_frame, old_gray;

    // Take first frame and find corners in it
    cap >> old_frame;
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    
    cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

    while (true) {
        cv::Mat frame, frame_gray;
        cap >> frame;
        if (frame.empty()) break;
        
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // Calculate Optical Flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        
        cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(15, 15), 2, criteria);

        std::vector<cv::Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++) {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // Draw the tracks
                cv::line(mask, p1[i], p0[i], cv::Scalar(0, 255, 0), 2);
                cv::circle(frame, p1[i], 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::Mat img;
        cv::add(frame, mask, img);

        cv::imshow("Lucas-Kanade Optical Flow", img);

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27) break;

        // Now update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
        
        // Re-detect if points are lost
        if (p0.size() < 10) {
             cv::goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
             mask = cv::Mat::zeros(old_frame.size(), old_frame.type());
        }
    }

    return 0;
}
