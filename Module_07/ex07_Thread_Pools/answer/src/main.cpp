#include <iostream>
#include <chrono>
#include "thread_pool.hpp"

int main() {
    cv_curriculum::ThreadPool pool(4);
    
    for(int i = 0; i < 8; ++i) {
        pool.enqueue([i] {
            std::cout << "Task " << i << " executing on " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });
    }
    
    // Give time for tasks to finish (simple test)
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    return 0;
}
