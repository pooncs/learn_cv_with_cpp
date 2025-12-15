#include <iostream>
#include <thread>
#include "resources.hpp"

int main(int argc, char** argv) {
    cv_curriculum::Resource r1, r2;
    
    std::cout << "Running safe swap..." << std::endl;
    // Should pass
    cv_curriculum::safeSwap(r1, r2);
    std::cout << "Done." << std::endl;
    
    return 0;
}
