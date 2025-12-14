#include <iostream>
#include <cassert>
#include <numeric>
#include "custom_filter.hpp"

void test_gaussian() {
    cv::Mat k = get_gaussian_kernel(3, 1.0);
    // Sum should be ~1.0
    double sum = cv::sum(k)[0];
    assert(std::abs(sum - 1.0) < 1e-4);
    std::cout << "[PASS] get_gaussian_kernel\n";
}

int main() {
    test_gaussian();
    return 0;
}
