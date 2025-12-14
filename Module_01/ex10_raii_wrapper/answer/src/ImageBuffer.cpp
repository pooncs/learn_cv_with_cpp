#include "ImageBuffer.hpp"
#include <iostream>
#include <cstring>
#include <utility>

ImageBuffer::ImageBuffer(int w, int h) : width(w), height(h) {
    data = new uint8_t[w * h];
    std::cout << "Alloc\n";
}

ImageBuffer::~ImageBuffer() {
    if (data) delete[] data;
    std::cout << "Free\n";
}

ImageBuffer::ImageBuffer(const ImageBuffer& other) : width(other.width), height(other.height) {
    data = new uint8_t[width * height];
    std::memcpy(data, other.data, width * height);
    std::cout << "Copy\n";
}

ImageBuffer& ImageBuffer::operator=(const ImageBuffer& other) {
    if (this == &other) return *this;
    delete[] data;
    width = other.width;
    height = other.height;
    data = new uint8_t[width * height];
    std::memcpy(data, other.data, width * height);
    return *this;
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept 
    : width(other.width), height(other.height), data(other.data) {
    other.data = nullptr;
    other.width = 0;
    other.height = 0;
    std::cout << "Move\n";
}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
    if (this == &other) return *this;
    delete[] data;
    width = other.width;
    height = other.height;
    data = other.data;
    other.data = nullptr;
    return *this;
}
