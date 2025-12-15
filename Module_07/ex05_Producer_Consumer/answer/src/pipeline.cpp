#include "pipeline.hpp"
#include <iostream>
#include <random>

namespace cv_curriculum {

void Pipeline::producer() {
    int frameId = 0;
    while (running) {
        // Simulate capture time (33ms = 30fps)
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        
        Frame f;
        f.id = frameId++;
        f.timestamp = std::chrono::steady_clock::now();
        // f.data...
        
        queue.push(std::move(f));
        // std::cout << "Produced Frame " << f.id << std::endl;
    }
    queue.stop();
}

void Pipeline::consumer() {
    Frame f;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 50); // Random processing time
    
    while (queue.pop(f)) {
        // Simulate processing
        int processTime = dist(rng);
        std::this_thread::sleep_for(std::chrono::milliseconds(processTime));
        
        // Log latency
        auto now = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - f.timestamp).count();
        
        std::cout << "Processed Frame " << f.id << " | Latency: " << latency << "ms" << std::endl;
    }
}

void Pipeline::start() {
    running = true;
    threads.emplace_back(&Pipeline::producer, this);
    threads.emplace_back(&Pipeline::consumer, this);
}

void Pipeline::stop() {
    running = false;
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    threads.clear();
}

} // namespace cv_curriculum
