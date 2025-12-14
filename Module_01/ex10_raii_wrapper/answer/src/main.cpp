#include "ImageBuffer.hpp"
#include <utility>

int main() {
    ImageBuffer img1(10, 10);
    ImageBuffer img2 = img1; // Copy
    ImageBuffer img3 = std::move(img1); // Move
    return 0;
}
