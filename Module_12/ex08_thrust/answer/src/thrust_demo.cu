#include "thrust_demo.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <algorithm>
#include <cmath>

struct SquareOp {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

void runThrustDemo() {
    int N = 1 << 20; // 1M elements
    std::cout << "Thrust Demo with " << N << " elements." << std::endl;

    // 1. Generate Data on Host
    thrust::host_vector<float> h_vec(N);
    std::generate(h_vec.begin(), h_vec.end(), []() { return (float)rand() / RAND_MAX; });

    // 2. Copy to Device
    thrust::device_vector<float> d_vec = h_vec;

    // 3. Sort
    std::cout << "Sorting..." << std::endl;
    thrust::sort(d_vec.begin(), d_vec.end());

    // 4. Transform (Square)
    std::cout << "Transforming (Square)..." << std::endl;
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), SquareOp());

    // 5. Reduce (Sum)
    std::cout << "Reducing..." << std::endl;
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    // 6. Verify
    // Copy back to verify sort order (of squared values)
    thrust::host_vector<float> h_result = d_vec;
    
    bool sorted = true;
    for(size_t i = 1; i < h_result.size(); ++i) {
        if(h_result[i] < h_result[i-1]) {
            sorted = false;
            std::cerr << "Sort failed at " << i << std::endl;
            break;
        }
    }

    // Verify sum roughly
    double cpu_sum = 0.0;
    for(float x : h_vec) {
        cpu_sum += x * x;
    }

    std::cout << "GPU Sum: " << sum << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << std::endl;

    if (sorted && std::abs(sum - cpu_sum) / cpu_sum < 1e-4) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }
}
