#include <iostream>
#include <vector>
#include "parallel_algos.hpp"

int main() {
    std::vector<double> data(100, 1.0);
    cv_curriculum::processParallel(data);
    
    // Stub check
    if (data[0] == 1.0) std::cout << "Did nothing (stub)." << std::endl;
    else std::cout << "Processed." << std::endl;
    
    return 0;
}
