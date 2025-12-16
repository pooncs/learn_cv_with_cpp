#include "factory.hpp"
#include <fmt/core.h>
#include <vector>

// Helper to register filters (normally done in cpp files)
void registerAll() {
    FilterFactory::registerFilter("Blur", []() { return std::make_unique<BlurFilter>(); });
    FilterFactory::registerFilter("Edge", []() { return std::make_unique<EdgeFilter>(); });
}

int main() {
    registerAll();

    // Imagine this comes from a config file
    std::vector<std::string> pipelineConfig = {"Blur", "Edge", "Blur"};

    // Create an image
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::circle(img, cv::Point(200, 200), 100, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, cv::Rect(50, 50, 100, 100), cv::Scalar(0, 0, 255), -1);

    fmt::print("Available Filters: \n");
    for (const auto& name : FilterFactory::getAvailableFilters()) {
        fmt::print(" - {}\n", name);
    }

    fmt::print("\nRunning Pipeline: {}\n", fmt::join(pipelineConfig, " -> "));

    cv::Mat currentImg = img.clone();
    
    // Process pipeline
    for (const auto& filterName : pipelineConfig) {
        try {
            auto filter = FilterFactory::createFilter(filterName);
            fmt::print("Applying {}...\n", filter->name());
            filter->process(currentImg);
        } catch (const std::exception& e) {
            fmt::print("Error: {}\n", e.what());
        }
    }

    // Visualization
    cv::imshow("Original", img);
    cv::imshow("Processed", currentImg);
    cv::waitKey(0);

    return 0;
}
