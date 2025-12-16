#include "image_processor.h"
#include <iostream>

ImageProcessor::ImageProcessor(const std::string& name) : m_name(name), m_isLoaded(false) {}

bool ImageProcessor::load(const std::string& path) {
    std::cout << "Loading image from " << path << std::endl;
    m_isLoaded = true;
    return true;
}

void ImageProcessor::applyBlur(int kernelSize, float sigma) {
    if (kernelSize % 2 == 0 || kernelSize <= 0) {
        throw std::invalid_argument("Kernel size must be odd and positive.");
    }
    std::cout << "Applying blur with kernel " << kernelSize << " and sigma " << sigma << std::endl;
}

std::string ImageProcessor::getStatus() const {
    return m_isLoaded ? "Image Loaded" : "Empty";
}

int main() {
    ImageProcessor proc("Demo");
    proc.load("test.jpg");
    proc.applyBlur(5, 1.0f);
    return 0;
}
