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
    
    cv_curriculum::SafeCounter safe;
    std::vector<std::thread> threads;
    for(int i=0; i<numThreads; ++i) threads.emplace_back(worker, std::ref(safe), iterations);
    for(auto& t : threads) t.join();
    
    if (safe.get() == numThreads * iterations) std::cout << "Success!" << std::endl;
    else std::cout << "Failure. Expected " << numThreads*iterations << " got " << safe.get() << std::endl;
    
    return 0;
}
