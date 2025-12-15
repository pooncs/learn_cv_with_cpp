#include <gtest/gtest.h>
#include <memory>

TEST(SharedPtrTest, Sharing) {
    auto p1 = std::make_shared<int>(100);
    EXPECT_EQ(p1.use_count(), 1);
    
    {
        auto p2 = p1;
        EXPECT_EQ(p1.use_count(), 2);
        EXPECT_EQ(*p2, 100);
    }
    
    EXPECT_EQ(p1.use_count(), 1);
}
