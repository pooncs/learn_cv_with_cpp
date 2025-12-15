#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Calibrator {
public:
    Calibrator(cv::Size board_size, float square_size);
    
    bool process_frame(const cv::Mat& frame, cv::Mat& display);
    void add_sample();
    double calibrate();
    void save(const std::string& filename);
    int get_sample_count() const;

private:
    cv::Size board_size_;
    float square_size_;
    cv::Size img_size_;
    
    // TODO: Add member variables for points and camera params
};
