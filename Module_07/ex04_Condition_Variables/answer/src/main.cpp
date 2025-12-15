#include <iostream>
#include <thread>
#include <vector>
#include "safe_queue.hpp"

void producer(cv_curriculum::SafeQueue<int>& q, int count) {
    for (int i = 0; i < count; ++i) {
        q.push(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    q.stop();
}

void consumer(cv_curriculum::SafeQueue<int>& q) {
    int item;
    while (q.pop(item)) {
        // Process item
        // std::cout << "Popped: " << item << std::endl;
    }
}

int main() {
    cv_curriculum::SafeQueue<int> q;
    int count = 1000;
    
    std::thread prod(producer, std::ref(q), count);
    std::thread cons(consumer, std::ref(q));
    
    prod.join();
    cons.join();
    
    std::cout << "Finished processing " << count << " items." << std::endl;
    return 0;
}
