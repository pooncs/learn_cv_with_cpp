#include <iostream>
#include "chessboard.hpp"

int main() {
    // Generate a synthetic chessboard
    cv::Size board_size(9, 6);
    int square_size = 40;
    cv::Mat img = cv::Mat::zeros(400, 600, CV_8UC1);
    img.setTo(255); // White background

    for (int i = 0; i < board_size.height + 1; ++i) {
        for (int j = 0; j < board_size.width + 1; ++j) {
            if ((i + j) % 2 == 0) {
                cv::Rect r(j * square_size + 50, i * square_size + 50, square_size, square_size);
                cv::rectangle(img, r, cv::Scalar(0), cv::FILLED);
            }
        }
    }

    std::vector<cv::Point2f> corners;
    bool found = detect_and_refine_corners(img, board_size, corners);

    if (found) {
        std::cout << "Found " << corners.size() << " corners.\n";
        cv::drawChessboardCorners(img, board_size, corners, found);
        // cv::imshow("Chessboard", img);
        // cv::waitKey(0);
    } else {
        std::cout << "Chessboard not found.\n";
    }

    return 0;
}
