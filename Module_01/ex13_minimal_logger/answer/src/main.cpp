#include <iostream>
#include <string>

// Task 1: Define Log Levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Task 2: The Logger Class
class Logger {
    static LogLevel current_level;

public:
    static void setLevel(LogLevel level) {
        current_level = level;
    }

    static void log(LogLevel level, const std::string& message) {
        // Task 3: Implementation
        if (level < current_level) {
            return; // Filter out messages below current level
        }

        std::string prefix;
        switch (level) {
            case LogLevel::DEBUG:   prefix = "[DEBUG] "; break;
            case LogLevel::INFO:    prefix = "[INFO]  "; break;
            case LogLevel::WARNING: prefix = "[WARN]  "; break;
            case LogLevel::ERROR:   prefix = "[ERROR] "; break;
        }

        // Use cerr for errors, cout for others (optional refinement)
        if (level == LogLevel::ERROR) {
            std::cerr << prefix << message << "\n";
        } else {
            std::cout << prefix << message << "\n";
        }
    }
};

// Initialize static member
LogLevel Logger::current_level = LogLevel::INFO;

int main() {
    std::cout << "--- Logger Test (Level: INFO) ---\n";
    Logger::setLevel(LogLevel::INFO);
    
    Logger::log(LogLevel::DEBUG, "This is a debug message (Hidden)");
    Logger::log(LogLevel::INFO, "System started.");
    Logger::log(LogLevel::WARNING, "Low battery.");
    Logger::log(LogLevel::ERROR, "Critical failure!");

    std::cout << "\n--- Logger Test (Level: DEBUG) ---\n";
    Logger::setLevel(LogLevel::DEBUG);
    Logger::log(LogLevel::DEBUG, "This is a debug message (Visible)");
    Logger::log(LogLevel::INFO, "System running.");

    return 0;
}
