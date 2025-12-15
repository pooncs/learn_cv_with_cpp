#include <gtest/gtest.h>
#include <string>
#include <utility>

class Movable {
public:
    std::string data;
    Movable(std::string s) : data(std::move(s)) {}
    Movable(Movable&& other) noexcept : data(std::move(other.data)) {
        other.data = "moved";
    }
};

TEST(MoveTest, MoveSemantics) {
    Movable a("content");
    Movable b(std::move(a));
    
    EXPECT_EQ(b.data, "content");
    EXPECT_EQ(a.data, "moved");
}
