#pragma once
#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string name_;
    // TODO: Add time_point member
};
