#include <iostream>
#include <cassert>
#include "mat_info.hpp"

void test_sharing() {
    cv::Mat A(100, 100, CV_8UC1);
    cv::Mat B = A;
    cv::Mat C = A.clone();

    assert(share_data(A, B) == true);
    assert(share_data(A, C) == false);
    std::cout << "[PASS] share_data\n";
}

int main() {
    test_sharing();
    return 0;
}
