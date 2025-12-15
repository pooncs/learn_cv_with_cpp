#include "async_tasks.hpp"
#include <thread>
#include <iostream>

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
    
    // Launch tasks
    // std::launch::async forces new thread (or thread pool execution)
    auto f_faces = std::async(std::launch::async, detectFaces, dummyInput);
    auto f_hist = std::async(std::launch::async, computeHistogram, dummyInput);
    auto f_save = std::async(std::launch::async, saveToDisk, dummyInput);
    
    ProcessResult res;
    // .get() blocks until ready
    res.faces = f_faces.get();
    res.histMean = f_hist.get();
    res.saved = f_save.get();
    
    auto end = std::chrono::high_resolution_clock::now();
    res.totalTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    return res;
}

} // namespace cv_curriculum
