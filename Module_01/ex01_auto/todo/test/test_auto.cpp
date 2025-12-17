#include <gtest/gtest.h>
#include "data_utils.hpp"
#include <type_traits>

TEST(AutoTest, VerifyDataStructure) {
    auto data = get_data();
    
    // Check if data has expected keys
    EXPECT_EQ(data.size(), 2);
    EXPECT_TRUE(data.find("ids") != data.end());
    EXPECT_TRUE(data.find("scores") != data.end());
    
    // Check values
    EXPECT_EQ(data["ids"].size(), 3);
    EXPECT_EQ(data["scores"].size(), 3);
    EXPECT_EQ(data["ids"][0], 1);
    EXPECT_EQ(data["scores"][0], 10);
}

TEST(AutoTest, VerifyTypes) {
    auto data = get_data();
    // Use decltype to inspect types
    using DataType = decltype(data);
    using ExpectedType = std::map<std::string, std::vector<int>>;
    
    // Static assertion to prove the types are identical
    static_assert(std::is_same_v<DataType, ExpectedType>, "Types should match!");
    
    SUCCEED();
}
