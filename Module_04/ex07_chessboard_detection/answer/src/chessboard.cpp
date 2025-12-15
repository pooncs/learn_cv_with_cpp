#include "chessboard.hpp"

bool detect_and_refine_corners(const cv::Mat& img, const cv::Size& board_size, std::vector<cv::Point2f>& corners) {
    bool found = cv::findChessboardCorners(img, board_size, corners, 
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img;
        }

        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    return found;
}
