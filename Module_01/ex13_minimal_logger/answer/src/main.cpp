#include "Logger.hpp"

int main() {
    LOG_INFO("System starting...");
    LOG_WARN("Low memory");
    LOG_ERROR("Connection failed");
    return 0;
}
