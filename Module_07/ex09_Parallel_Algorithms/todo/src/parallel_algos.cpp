#include "parallel_algos.hpp"
#include <cmath>
#include <algorithm>
#include <execution>

namespace cv_curriculum {

double heavyComputation(double x) {
    return std::sin(x) * std::cos(x) + std::sqrt(std::abs(x));
}

void processSequential(std::vector<double>& data) {
    // TODO: Use std::transform with std::execution::seq
}

void processParallel(std::vector<double>& data) {
    // TODO: Use std::transform with std::execution::par
}

} // namespace cv_curriculum
