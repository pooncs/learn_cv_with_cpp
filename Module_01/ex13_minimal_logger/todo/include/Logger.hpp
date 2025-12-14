#pragma once
#include <iostream>
#include <string>

enum class LogLevel { INFO, WARN, ERROR, DEBUG };

class Logger {
public:
    static Logger& instance();

    void log(LogLevel level, const std::string& msg);

private:
    Logger() = default;
};

// TODO: Define macros LOG_INFO, LOG_WARN, LOG_ERROR
