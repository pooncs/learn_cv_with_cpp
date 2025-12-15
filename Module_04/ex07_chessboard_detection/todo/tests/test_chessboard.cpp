#include <gtest/gtest.h>
#include "chessboard.hpp"

TEST(ChessboardTest, Detection) {
    cv::Size board_size(3, 3); // 3x3 inner corners
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC1);
    img.setTo(255);
    
    // Draw 4x4 squares
    int sz = 40;
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            if((i+j)%2 == 0) {
                cv::Rect r(j*sz + 20, i*sz + 20, sz, sz);
                cv::rectangle(img, r, cv::Scalar(0), cv::FILLED);
            }
        }
    }
    
    std::vector<cv::Point2f> corners;
    bool found = detect_and_refine_corners(img, board_size, corners);
    
    EXPECT_TRUE(found);
    if(found) {
        EXPECT_EQ(corners.size(), 9);
    }
}
