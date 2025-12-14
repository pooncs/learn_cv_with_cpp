#pragma once
#include <iostream>
#include <string>
#include <mutex>

enum class LogLevel { INFO, WARN, ERROR, DEBUG };

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        switch (level) {
            case LogLevel::INFO:  std::cout << "[INFO] "; break;
            case LogLevel::WARN:  std::cout << "[WARN] "; break;
            case LogLevel::ERROR: std::cerr << "[ERROR] "; break;
            case LogLevel::DEBUG: std::cout << "[DEBUG] "; break;
        }
        std::cout << msg << std::endl;
    }

private:
    Logger() = default;
    std::mutex mutex_;
};

#define LOG_INFO(msg) Logger::instance().log(LogLevel::INFO, msg)
#define LOG_WARN(msg) Logger::instance().log(LogLevel::WARN, msg)
#define LOG_ERROR(msg) Logger::instance().log(LogLevel::ERROR, msg)
