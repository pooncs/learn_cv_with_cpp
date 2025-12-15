#include <gtest/gtest.h>
#include "ImageBuffer.hpp"

TEST(RAIITest, BufferManagement) {
    {
        ImageBuffer buf(100);
        EXPECT_NE(buf.data(), nullptr);
        EXPECT_EQ(buf.size(), 100);
    }
    // Destructor called, memory freed (cannot easily verify here without mocking)
}
