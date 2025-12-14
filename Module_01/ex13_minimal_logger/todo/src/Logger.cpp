#include "Logger.hpp"

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::log(LogLevel level, const std::string& msg) {
    // TODO: Implement logging logic (switch case on level, print to cout/cerr)
}
