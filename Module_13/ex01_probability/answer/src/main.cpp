#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include "probability.h"

void printBelief(const std::vector<double>& b) {
    std::cout << "[";
    for (size_t i = 0; i < b.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << b[i];
        if (i < b.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // 1. Setup
    int N = 20;
    std::vector<double> belief(N, 0.0);
    belief[10] = 1.0; // Certainty at pos 10

    std::vector<double> kernel = {0.1, 0.8, 0.1}; // Move 1 step with noise

    std::cout << "Initial Belief:" << std::endl;
    printBelief(belief);

    // 2. Predict (Move)
    std::cout << "\nAfter Step 1 (Convolution):" << std::endl;
    belief = convolve1D(belief, kernel);
    printBelief(belief);

    // 3. Predict (Move)
    std::cout << "\nAfter Step 2 (Convolution):" << std::endl;
    belief = convolve1D(belief, kernel);
    printBelief(belief);
    
    // 4. Predict (Move)
    std::cout << "\nAfter Step 3 (Convolution):" << std::endl;
    belief = convolve1D(belief, kernel);
    printBelief(belief);

    // Verify Sum
    double sum = std::accumulate(belief.begin(), belief.end(), 0.0);
    std::cout << "\nSum: " << sum << std::endl;

    if (abs(sum - 1.0) < 1e-5) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    return 0;
}
