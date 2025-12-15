#pragma once
#include <thread>
#include <chrono>

inline void process_image(int w, int h) {
    // Simulate work: 1ms
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
