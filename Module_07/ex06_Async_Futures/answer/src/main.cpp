#include <iostream>
#include "async_tasks.hpp"

int main() {
    std::cout << "Starting async processing..." << std::endl;
    
    // Expected sequential time: 50 + 20 + 80 = 150ms
    // Expected parallel time: max(50, 20, 80) = 80ms (plus overhead)
    
    auto res = cv_curriculum::processFrameAsync(10);
    
    std::cout << "Result: " << std::endl;
    std::cout << "  Faces: " << res.faces << std::endl;
    std::cout << "  Hist:  " << res.histMean << std::endl;
    std::cout << "  Saved: " << res.saved << std::endl;
    std::cout << "  Time:  " << res.totalTimeMs << " ms" << std::endl;
    
    if (res.totalTimeMs < 140.0) {
        std::cout << "Success! Time is less than sequential sum." << std::endl;
    } else {
        std::cout << "Warning: Time is close to sequential sum." << std::endl;
    }
    
    return 0;
}
