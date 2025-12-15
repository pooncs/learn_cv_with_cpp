#include "optimizer.hpp"
#include <fmt/core.h>
#include <random>

int main() {
    // True params: a=1.0, b=2.0, c=1.0
    double a_true = 1.0, b_true = 2.0, c_true = 1.0;
    
    std::vector<DataPoint> data;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);

    for (double x = -2.0; x <= 2.0; x += 0.1) {
        double y = a_true * x * x + b_true * x + c_true + noise(rng);
        data.push_back({x, y});
    }

    fmt::print("Starting optimization with {} points...\n", data.size());

    auto result = CurveFitter::solve(data);

    fmt::print("True params: a={:.2f}, b={:.2f}, c={:.2f}\n", a_true, b_true, c_true);
    fmt::print("Estimated:   a={:.2f}, b={:.2f}, c={:.2f}\n", result[0], result[1], result[2]);

    return 0;
}
