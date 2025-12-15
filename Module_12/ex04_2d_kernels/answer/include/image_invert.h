#pragma once

// Inverts image colors. Assumes 3 channels (RGB/BGR)
void invertImageWrapper(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
