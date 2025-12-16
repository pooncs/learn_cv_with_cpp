#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// Abstract Interface
class IFilter {
public:
    virtual ~IFilter() = default;
    virtual void process(cv::Mat& img) = 0;
    virtual std::string name() const = 0;
};

// Concrete Implementation 1
class BlurFilter : public IFilter {
public:
    void process(cv::Mat& img) override {
        cv::GaussianBlur(img, img, cv::Size(15, 15), 0);
    }
    std::string name() const override { return "Blur"; }
};

// Concrete Implementation 2
class EdgeFilter : public IFilter {
public:
    void process(cv::Mat& img) override {
        cv::Canny(img, img, 100, 200);
        // Canny returns single channel, ensure we keep it consistent if needed, 
        // but for this example it's fine.
        if (img.channels() == 1) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }
    }
    std::string name() const override { return "Edge"; }
};
