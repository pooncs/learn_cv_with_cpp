#pragma once
#include <future>
#include <vector>

namespace cv_curriculum {

// Simulation functions
int detectFaces(int dummyInput);
float computeHistogram(int dummyInput);
bool saveToDisk(int dummyInput);

struct ProcessResult {
    int faces;
    float histMean;
    bool saved;
    double totalTimeMs;
};

// Function to run them in parallel
ProcessResult processFrameAsync(int dummyInput);

} // namespace cv_curriculum
