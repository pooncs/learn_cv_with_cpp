#include "factory.hpp"
#include <fmt/core.h>
#include <vector>
#include <opencv2/opencv.hpp>

// --- Define IFilter and Concrete Filters Here or in Headers ---
class IFilter {
public:
    virtual ~IFilter() = default;
    virtual void process(cv::Mat& img) = 0;
    virtual std::string name() const = 0;
};

// TODO: Implement BlurFilter : public IFilter
// TODO: Implement EdgeFilter : public IFilter

int main() {
    // TODO: Register your filters
    // FilterFactory::registerFilter("Blur", ...);

    std::vector<std::string> pipelineConfig = {"Blur", "Edge", "Blur"};
    
    cv::Mat img = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::circle(img, cv::Point(200, 200), 100, cv::Scalar(255, 255, 255), -1);

    fmt::print("Running Pipeline...\n");

    // TODO: Iterate through pipelineConfig, create filters using factory, and process image.

    return 0;
}
