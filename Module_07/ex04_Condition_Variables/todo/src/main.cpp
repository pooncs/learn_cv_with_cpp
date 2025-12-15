#include <iostream>
#include <thread>
#include "safe_queue.hpp"

int main() {
    cv_curriculum::SafeQueue<int> q;
    
    // Stub test
    q.push(1);
    int item;
    if (q.pop(item)) {
        std::cout << "Popped " << item << std::endl;
    } else {
        std::cout << "Failed to pop (Function not implemented)" << std::endl;
    }
    
    return 0;
}
