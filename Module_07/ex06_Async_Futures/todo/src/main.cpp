#include <iostream>
#include "async_tasks.hpp"

int main() {
    auto res = cv_curriculum::processFrameAsync(10);
    std::cout << "Time: " << res.totalTimeMs << " ms" << std::endl;
    return 0;
}
