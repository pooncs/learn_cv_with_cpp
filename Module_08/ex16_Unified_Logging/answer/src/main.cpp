#include "logger.hpp"

int main(int argc, char** argv) {
    init_logging(argv[0]);
    
    LOG_INFO("Application started with {} arguments", argc);
    LOG_WARN("This is a warning from unified logger");
    
    return 0;
}
