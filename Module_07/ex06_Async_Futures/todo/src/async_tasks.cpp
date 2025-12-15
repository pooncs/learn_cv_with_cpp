#include "async_tasks.hpp"
#include <thread>
#include <chrono>

namespace cv_curriculum {

int detectFaces(int dummyInput) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return dummyInput % 5;
}

float computeHistogram(int dummyInput) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return dummyInput * 1.5f;
}

bool saveToDisk(int dummyInput) {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    return true;
}

ProcessResult processFrameAsync(int dummyInput) {
    auto start = std::chrono::high_resolution_clock::now();
    ProcessResult res;
    
    // TODO: Launch tasks using std::async
    // auto f1 = ...
    // auto f2 = ...
    
    // TODO: Collect results using .get()
    
    // Sequential fallback (remove this when implemented)
    res.faces = detectFaces(dummyInput);
    res.histMean = computeHistogram(dummyInput);
    res.saved = saveToDisk(dummyInput);
    
    auto end = std::chrono::high_resolution_clock::now();
    res.totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    return res;
}

} // namespace cv_curriculum
