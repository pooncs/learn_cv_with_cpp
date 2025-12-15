#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Calibrator {
public:
    Calibrator(cv::Size board_size, float square_size);
    
    // Returns true if pattern found
    bool process_frame(const cv::Mat& frame, cv::Mat& display);
    
    // Add current detection to calibration set
    void add_sample();
    
    // Run calibration
    double calibrate();
    
    // Save to file
    void save(const std::string& filename);

    int get_sample_count() const { return object_points_.size(); }

private:
    cv::Size board_size_;
    float square_size_;
    cv::Size img_size_;
    
    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;
    
    std::vector<cv::Point2f> current_corners_;
    bool current_found_ = false;
    
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
};
