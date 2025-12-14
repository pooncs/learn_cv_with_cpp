#include "ImageBuffer.hpp"
#include <iostream>

ImageBuffer::ImageBuffer(int w, int h) : width(w), height(h) {
    data = new uint8_t[w * h];
    std::cout << "Alloc\n";
}

ImageBuffer::~ImageBuffer() {
    delete[] data;
    std::cout << "Free\n";
}

// TODO: Implement Rule of Five
