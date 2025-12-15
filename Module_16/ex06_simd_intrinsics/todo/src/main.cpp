#include "simd_ops.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<float> a(16, 1.0f);
    std::vector<float> b(16, 2.0f);

    float res_scalar = dot_product_scalar(a.data(), b.data(), a.size());
    float res_avx = dot_product_avx2(a.data(), b.data(), a.size());

    std::cout << "Scalar: " << res_scalar << "\n";
    std::cout << "AVX2: " << res_avx << "\n";
    
    if (res_scalar == res_avx) {
        std::cout << "Matches!" << std::endl;
    } else {
        std::cout << "Mismatch!" << std::endl;
    }
    
    std::cout << "TODO: Implement proper AVX2 logic and run benchmarks." << std::endl;
    return 0;
}
