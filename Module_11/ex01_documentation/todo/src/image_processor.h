#pragma once
#include <vector>
#include <string>

// TODO: Add Doxygen class description
class ImageProcessor {
public:
    // TODO: Add Constructor documentation
    ImageProcessor(const std::string& name);

    // TODO: Add load method documentation (@param, @return)
    bool load(const std::string& path);

    // TODO: Add applyBlur documentation (@param, @note about kernel size)
    void applyBlur(int kernelSize, float sigma);

    // TODO: Add getStatus documentation
    std::string getStatus() const;

private:
    std::string m_name; 
    bool m_isLoaded;    
};
