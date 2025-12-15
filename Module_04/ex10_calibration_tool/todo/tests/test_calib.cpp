#include <gtest/gtest.h>
#include "Calibrator.hpp"

TEST(CalibratorTest, Init) {
    Calibrator c(cv::Size(9, 6), 0.025f);
    EXPECT_EQ(c.get_sample_count(), 0);
}
