#include <gtest/gtest.h>
#include <memory>

TEST(UniquePtrTest, Ownership) {
    std::unique_ptr<int> p1 = std::make_unique<int>(42);
    EXPECT_EQ(*p1, 42);
    
    std::unique_ptr<int> p2 = std::move(p1);
    EXPECT_EQ(*p2, 42);
    EXPECT_EQ(p1, nullptr);
}
