#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <QImage>

TEST(ImageDisplay, Conversion) {
    // Create a simple red image (BGR: 0, 0, 255)
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    
    // Convert to RGB
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    
    // Check pixel at 0,0 is now RGB(255, 0, 0)
    cv::Vec3b p = mat.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(p[0], 255);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 0);

    // Create QImage
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    
    // Verify QImage properties
    EXPECT_EQ(qimg.width(), 100);
    EXPECT_EQ(qimg.height(), 100);
    EXPECT_EQ(qimg.format(), QImage::Format_RGB888);
    
    // Verify pixel color in QImage
    QColor c = qimg.pixelColor(0, 0);
    EXPECT_EQ(c.red(), 255);
    EXPECT_EQ(c.green(), 0);
    EXPECT_EQ(c.blue(), 0);
}
