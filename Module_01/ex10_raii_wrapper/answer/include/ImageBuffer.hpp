#pragma once
#include <cstdint>
#include <algorithm>

class ImageBuffer {
public:
    ImageBuffer(int w, int h);
    ~ImageBuffer();

    ImageBuffer(const ImageBuffer& other);
    ImageBuffer& operator=(const ImageBuffer& other);

    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

private:
    int width, height;
    uint8_t* data;
};
