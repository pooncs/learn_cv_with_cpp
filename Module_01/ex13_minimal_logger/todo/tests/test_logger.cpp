#include <gtest/gtest.h>
#include "Logger.hpp"

TEST(LoggerTest, Compiles) {
    Logger::instance().log(LogLevel::INFO, "Test message");
    SUCCEED();
}
