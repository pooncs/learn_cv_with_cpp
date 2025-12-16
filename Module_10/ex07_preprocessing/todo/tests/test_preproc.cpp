#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// Mock preprocess
std::vector<float> preprocess(const cv::Mat& input_img, int target_w, int target_h) {
    cv::Mat img;
    cv::resize(input_img, img, cv::Size(target_w, target_h));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    std::vector<float> output;
    for (int c = 0; c < 3; ++c) {
        output.insert(output.end(), (float*)channels[c].data, (float*)channels[c].data + target_w * target_h);
    }
    return output;
}

TEST(Preprocessing, OutputShape) {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
    auto res = preprocess(img, 10, 10);
    
    // Size should be 3 * 10 * 10 = 300
    EXPECT_EQ(res.size(), 300);
    
    // First channel is R. Original was BGR(255,0,0) -> RGB(0,0,255).
    // Wait, BGR(255,0,0) means Blue=255.
    // RGB conversion -> R=0, G=0, B=255.
    // Channel 0 (R) should be 0.
    EXPECT_NEAR(res[0], 0.0f, 1e-5);
    
    // Channel 2 (B) start index = 10*10*2 = 200. Should be 1.0.
    EXPECT_NEAR(res[200], 1.0f, 1e-5);
}
