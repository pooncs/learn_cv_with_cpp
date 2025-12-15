#include "Calibrator.hpp"

Calibrator::Calibrator(cv::Size board_size, float square_size) 
    : board_size_(board_size), square_size_(square_size) 
{
}

bool Calibrator::process_frame(const cv::Mat& frame, cv::Mat& display) {
    // TODO: Detect corners, draw them
    return false;
}

void Calibrator::add_sample() {
    // TODO: Store object points and image points
}

double Calibrator::calibrate() {
    // TODO: Run calibrateCamera
    return 0.0;
}

void Calibrator::save(const std::string& filename) {
    // TODO: Save K and D using cv::FileStorage
}

int Calibrator::get_sample_count() const {
    return 0;
}
