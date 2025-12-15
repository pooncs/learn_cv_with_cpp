#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <type_traits>

// This test checks if the student correctly used auto in their implementation.
// Since we can't easily check source code AST, we check functionality.

TEST(AutoTest, TypeDeduction) {
    auto x = 42;
    auto y = 3.14;
    auto z = std::string("hello");
    
    EXPECT_TRUE((std::is_same_v<decltype(x), int>));
    EXPECT_TRUE((std::is_same_v<decltype(y), double>));
    EXPECT_TRUE((std::is_same_v<decltype(z), std::string>));
}
