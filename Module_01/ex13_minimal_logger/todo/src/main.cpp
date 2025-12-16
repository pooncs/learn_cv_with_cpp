#include <iostream>
#include <string>

// Task 1: Define Log Levels
// enum class LogLevel { ... };

// Task 2: The Logger Class
class Logger {
    // static LogLevel current_level;

public:
    // static void setLevel(LogLevel level) { ... }
    
    // static void log(LogLevel level, const std::string& message) { ... }
};

// Initialize static member
// LogLevel Logger::current_level = LogLevel::INFO;

int main() {
    std::cout << "--- Logger Test ---\n";

    // Task 4: Usage
    // Logger::setLevel(LogLevel::INFO);
    
    // Logger::log(LogLevel::DEBUG, "This is a debug message (Hidden)");
    // Logger::log(LogLevel::INFO, "System started.");
    // Logger::log(LogLevel::ERROR, "Critical failure!");

    // Logger::setLevel(LogLevel::DEBUG);
    // Logger::log(LogLevel::DEBUG, "This is a debug message (Visible)");

    return 0;
}
