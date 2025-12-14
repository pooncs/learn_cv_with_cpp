#include <gtest/gtest.h>
#include "FileIO.hpp"
#include <cstdio>

TEST(FileIOTest, Binary) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    save_binary("test.bin", data);
    auto loaded = load_binary("test.bin");
    ASSERT_EQ(loaded.size(), data.size());
    EXPECT_FLOAT_EQ(loaded[0], data[0]);
    std::remove("test.bin");
}

TEST(FileIOTest, Json) {
    Config cfg{100, 200, "Test"};
    save_config("test.json", cfg);
    auto loaded = load_config("test.json");
    EXPECT_EQ(loaded.width, 100);
    EXPECT_EQ(loaded.app_name, "Test");
    std::remove("test.json");
}
