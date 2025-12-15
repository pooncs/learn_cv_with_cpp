#include <iostream>
#include "thread_pool.hpp"

int main() {
    cv_curriculum::ThreadPool pool(2);
    
    pool.enqueue([]{ std::cout << "Hello" << std::endl; });
    
    // Stub might crash or do nothing
    return 0;
}
