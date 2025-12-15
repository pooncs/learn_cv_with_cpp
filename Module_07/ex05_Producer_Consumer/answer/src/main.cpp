#include <iostream>
#include <thread>
#include "pipeline.hpp"

int main() {
    cv_curriculum::Pipeline pipeline;
    
    std::cout << "Starting pipeline..." << std::endl;
    pipeline.start();
    
    // Run for 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    std::cout << "Stopping pipeline..." << std::endl;
    pipeline.stop();
    
    return 0;
}
