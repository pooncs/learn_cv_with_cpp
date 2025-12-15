#include <gtest/gtest.h>
#include <variant>
#include <string>

TEST(VariantTest, BasicUsage) {
    std::variant<int, std::string> v;
    v = 42;
    EXPECT_TRUE(std::holds_alternative<int>(v));
    EXPECT_EQ(std::get<int>(v), 42);
    
    v = "hello";
    EXPECT_TRUE(std::holds_alternative<std::string>(v));
    EXPECT_EQ(std::get<std::string>(v), "hello");
}
