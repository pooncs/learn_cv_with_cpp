#include "ImageBuffer.hpp"
#include <utility>

int main() {
    ImageBuffer img1(10, 10);
    // ImageBuffer img2 = img1; // Should work after implementing copy
    // ImageBuffer img3 = std::move(img1); // Should work after implementing move
    return 0;
}
