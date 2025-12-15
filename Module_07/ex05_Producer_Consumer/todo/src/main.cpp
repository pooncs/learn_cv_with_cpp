#include <iostream>
#include <thread>
#include "pipeline.hpp"

int main() {
    std::cout << "Pipeline Exercise" << std::endl;
    cv_curriculum::Pipeline pipeline;
    
    pipeline.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pipeline.stop();
    
    std::cout << "Done." << std::endl;
    return 0;
}
