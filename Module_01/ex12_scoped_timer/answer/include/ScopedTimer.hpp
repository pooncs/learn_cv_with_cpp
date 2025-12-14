#pragma once
#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << "[Timer] " << name_ << ": " << duration << " us\n";
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
