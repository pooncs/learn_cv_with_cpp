#include <iostream>
#include <thread>
#include <vector>
#include "atomic_counter.hpp"

int main() {
    cv_curriculum::AtomicCounter counter;
    int numThreads = 10;
    int iterations = 100000;
    
    std::vector<std::thread> threads;
    for(int i=0; i<numThreads; ++i) {
        threads.emplace_back([&counter, iterations](){
            for(int j=0; j<iterations; ++j) {
                counter.increment();
            }
        });
    }
    
    for(auto& t : threads) t.join();
    
    std::cout << "Count: " << counter.get() << std::endl;
    std::cout << "Expected: " << numThreads * iterations << std::endl;
    
    if (counter.get() == numThreads * iterations) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }
    
    return 0;
}
