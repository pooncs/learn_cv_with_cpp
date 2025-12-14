#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include "ScopedTimer.hpp"

void heavy_computation() {
    ScopedTimer t("Heavy Computation");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int main() {
    std::cout << "Starting timer demo...\n";
    {
        ScopedTimer t("Main Block");
        heavy_computation();
    }
    return 0;
}
