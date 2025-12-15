#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include "buffer_pool.hpp"

void worker(cv_curriculum::BufferPool& pool, int id) {
    // Acquire
    std::cout << "Thread " << id << " waiting for buffer..." << std::endl;
    auto handle = pool.acquire();
    std::cout << "Thread " << id << " got buffer." << std::endl;
    
    // Use
    cv::Mat& mat = handle.get();
    cv::randn(mat, 128, 50); // Simulate write
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Simulate processing
    
    // Release (auto via destructor)
    std::cout << "Thread " << id << " releasing buffer." << std::endl;
}

int main() {
    // Pool of size 2
    cv_curriculum::BufferPool pool(2, 1000, 1000, CV_8UC1);
    
    // Launch 4 threads
    std::vector<std::thread> threads;
    for(int i=0; i<4; ++i) {
        threads.emplace_back(worker, std::ref(pool), i);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Stagger slightly
    }
    
    for(auto& t : threads) t.join();
    
    std::cout << "All done." << std::endl;
    return 0;
}
