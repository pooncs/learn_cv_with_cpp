#pragma once
#include <vector>
#include <functional>

namespace cv_curriculum {

// Operation to apply to each element
double heavyComputation(double x);

void processSequential(std::vector<double>& data);
void processParallel(std::vector<double>& data);

} // namespace cv_curriculum
