#include "Calibrator.hpp"

Calibrator::Calibrator(cv::Size board_size, float square_size) 
    : board_size_(board_size), square_size_(square_size) 
{
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
}

bool Calibrator::process_frame(const cv::Mat& frame, cv::Mat& display) {
    img_size_ = frame.size();
    frame.copyTo(display);
    
    bool found = cv::findChessboardCorners(frame, board_size_, current_corners_, 
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        
    if (found) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(gray, current_corners_, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            
        cv::drawChessboardCorners(display, board_size_, current_corners_, found);
    }
    current_found_ = found;
    return found;
}

void Calibrator::add_sample() {
    if (!current_found_) return;
    
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < board_size_.height; ++i) {
        for (int j = 0; j < board_size_.width; ++j) {
            obj.emplace_back(j * square_size_, i * square_size_, 0);
        }
    }
    
    object_points_.push_back(obj);
    image_points_.push_back(current_corners_);
}

double Calibrator::calibrate() {
    if (object_points_.empty()) return -1;
    
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points_, image_points_, img_size_,
        camera_matrix_, dist_coeffs_, rvecs, tvecs);
        
    return rms;
}

void Calibrator::save(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix_;
    fs << "dist_coeffs" << dist_coeffs_;
    fs.release();
}
