#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> preprocess(const cv::Mat& input_img, int target_w, int target_h) {
    cv::Mat img;
    
    // 1. Resize
    cv::resize(input_img, img, cv::Size(target_w, target_h));

    // 2. Convert Color (BGR -> RGB)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 3. Normalize (0-255 -> 0.0-1.0)
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // 4. HWC -> NCHW
    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    std::vector<float> output;
    output.reserve(3 * target_w * target_h);

    for (int c = 0; c < 3; ++c) {
        // Append flat channel data
        output.insert(output.end(), (float*)channels[c].data, (float*)channels[c].data + target_w * target_h);
    }
    
    return output;
}

int main() {
    cv::Mat img = cv::imread("data/lenna.png");
    if (img.empty()) {
        img = cv::Mat(400, 600, CV_8UC3, cv::Scalar(0, 255, 0)); // Dummy
    }

    auto data = preprocess(img, 224, 224);

    std::cout << "Data size: " << data.size() << std::endl;
    std::cout << "First 10 values: ";
    for(int i=0; i<10; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;

    return 0;
}
