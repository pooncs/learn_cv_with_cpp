#include <iostream>
#include "buffer_pool.hpp"

int main() {
    cv_curriculum::BufferPool pool(1, 100, 100, CV_8UC1);
    auto h = pool.acquire();
    // Stub check
    return 0;
}
