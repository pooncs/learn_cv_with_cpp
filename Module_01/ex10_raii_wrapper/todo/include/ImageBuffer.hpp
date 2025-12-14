#pragma once
#include <cstdint>

class ImageBuffer {
public:
    ImageBuffer(int w, int h);
    ~ImageBuffer();

    // TODO: Implement Rule of Five
    // ImageBuffer(const ImageBuffer& other);
    // ImageBuffer& operator=(const ImageBuffer& other);
    // ImageBuffer(ImageBuffer&& other) noexcept;
    // ImageBuffer& operator=(ImageBuffer&& other) noexcept;

private:
    int width, height;
    uint8_t* data;
};
