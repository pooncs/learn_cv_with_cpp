#include <iostream>
#include <vector>
#include <thread>
#include "counters.hpp"

void worker(cv_curriculum::Counter& c, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        c.increment();
    }
}

int main() {
    int numThreads = 10;
    int iterations = 10000;
    int expected = numThreads * iterations;
    
    // Unsafe
    cv_curriculum::UnsafeCounter unsafe;
    std::vector<std::thread> threads;
    for(int i=0; i<numThreads; ++i) threads.emplace_back(worker, std::ref(unsafe), iterations);
    for(auto& t : threads) t.join();
    
    std::cout << "Unsafe Result: " << unsafe.get() << " (Expected: " << expected << ")" << std::endl;
    if (unsafe.get() != expected) std::cout << "Race detected!" << std::endl;
    else std::cout << "Lucky run (no race detected, but it exists)." << std::endl;
    
    // Safe
    cv_curriculum::SafeCounter safe;
    threads.clear();
    for(int i=0; i<numThreads; ++i) threads.emplace_back(worker, std::ref(safe), iterations);
    for(auto& t : threads) t.join();
    
    std::cout << "Safe Result:   " << safe.get() << " (Expected: " << expected << ")" << std::endl;
    
    return 0;
}
