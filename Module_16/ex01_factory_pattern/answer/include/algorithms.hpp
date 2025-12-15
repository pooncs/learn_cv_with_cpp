#pragma once
#include <string>
#include <fmt/core.h>

// Mock OpenCV Mat
struct Mat {
    std::string data;
};

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;
    virtual void process(const Mat& input) = 0;
    virtual std::string name() const = 0;
};

class CannyDetector : public IAlgorithm {
public:
    void process(const Mat& input) override {
        fmt::print("[Canny] Processing {}\n", input.data);
    }
    std::string name() const override { return "Canny"; }
};

class SobelDetector : public IAlgorithm {
public:
    void process(const Mat& input) override {
        fmt::print("[Sobel] Processing {}\n", input.data);
    }
    std::string name() const override { return "Sobel"; }
};
