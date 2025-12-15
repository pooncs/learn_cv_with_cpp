#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "parallel_algos.hpp"

int main() {
    size_t size = 10000000; // 10 Million
    std::cout << "Generating " << size << " elements..." << std::endl;
    
    std::vector<double> data(size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    for(auto& x : data) x = dist(rng);
    
    std::vector<double> dataSeq = data;
    std::vector<double> dataPar = data;
    
    // Sequential
    auto start = std::chrono::high_resolution_clock::now();
    cv_curriculum::processSequential(dataSeq);
    auto end = std::chrono::high_resolution_clock::now();
    double timeSeq = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sequential: " << timeSeq << " ms" << std::endl;
    
    // Parallel
    start = std::chrono::high_resolution_clock::now();
    cv_curriculum::processParallel(dataPar);
    end = std::chrono::high_resolution_clock::now();
    double timePar = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel:   " << timePar << " ms" << std::endl;
    
    if (timePar < timeSeq) {
        std::cout << "Speedup: " << timeSeq / timePar << "x" << std::endl;
    } else {
        std::cout << "No speedup (overhead might dominate or not supported)." << std::endl;
    }
    
    // Verify
    if (dataSeq == dataPar) {
        std::cout << "Results match." << std::endl;
    } else {
        std::cout << "Results differ!" << std::endl;
    }
    
    return 0;
}
