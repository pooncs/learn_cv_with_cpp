#include <iostream>
#include <thread>
#include "resources.hpp"

int main(int argc, char** argv) {
    cv_curriculum::Resource r1, r2;
    r1.data = 1;
    r2.data = 2;
    
    // Test Safe Swap
    std::thread t1(cv_curriculum::safeSwap, std::ref(r1), std::ref(r2));
    std::thread t2(cv_curriculum::safeSwap, std::ref(r2), std::ref(r1));
    
    t1.join();
    t2.join();
    
    std::cout << "Safe swap finished. r1=" << r1.data << ", r2=" << r2.data << std::endl;
    
    // Uncomment to test deadlock (might hang indefinitely)
    /*
    std::cout << "Starting unsafe swap..." << std::endl;
    std::thread t3(cv_curriculum::unsafeSwap, std::ref(r1), std::ref(r2));
    std::thread t4(cv_curriculum::unsafeSwap, std::ref(r2), std::ref(r1));
    t3.join();
    t4.join();
    std::cout << "Unsafe swap finished (lucky if you see this)." << std::endl;
    */
    
    return 0;
}
