#include "parallel_algos.hpp"
#include <cmath>
#include <algorithm>
#include <execution>

namespace cv_curriculum {

double heavyComputation(double x) {
    // Simulate complex pixel operation
    return std::sin(x) * std::cos(x) + std::sqrt(std::abs(x));
}

void processSequential(std::vector<double>& data) {
    std::transform(std::execution::seq, data.begin(), data.end(), data.begin(), heavyComputation);
}

void processParallel(std::vector<double>& data) {
    std::transform(std::execution::par, data.begin(), data.end(), data.begin(), heavyComputation);
}

} // namespace cv_curriculum
