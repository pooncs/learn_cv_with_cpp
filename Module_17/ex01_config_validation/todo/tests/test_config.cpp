#include <gtest/gtest.h>
#include "config.hpp"
#include <fstream>

// Helper to write temp config
void write_config(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
}

TEST(ConfigTest, ValidConfig) {
    std::string content = R"(
model:
  name: "TestModel"
  input_size: [224, 224]
  confidence_threshold: 0.8
dataset:
  path: "/tmp/data"
)";
    write_config("test_valid.yaml", content);
    AppConfig config = ConfigLoader::load("test_valid.yaml");
    EXPECT_EQ(config.model.name, "TestModel");
    EXPECT_EQ(config.model.input_size[0], 224);
    EXPECT_FLOAT_EQ(config.model.confidence_threshold, 0.8f);
}

TEST(ConfigTest, MissingModel) {
    std::string content = R"(
dataset:
  path: "/tmp/data"
)";
    write_config("test_missing_model.yaml", content);
    EXPECT_THROW(ConfigLoader::load("test_missing_model.yaml"), std::runtime_error);
}

TEST(ConfigTest, InvalidConfidence) {
    std::string content = R"(
model:
  name: "TestModel"
  input_size: [224, 224]
  confidence_threshold: 1.5
dataset:
  path: "/tmp/data"
)";
    write_config("test_invalid_conf.yaml", content);
    EXPECT_THROW(ConfigLoader::load("test_invalid_conf.yaml"), std::runtime_error);
}
