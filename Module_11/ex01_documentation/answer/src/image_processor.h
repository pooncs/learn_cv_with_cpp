#pragma once
#include <vector>
#include <string>

/**
 * @brief A class for processing images.
 * 
 * This class demonstrates standard Doxygen documentation patterns.
 * It simulates basic image operations.
 */
class ImageProcessor {
public:
    /**
     * @brief Construct a new Image Processor object.
     * 
     * @param name The name of the processor instance.
     */
    ImageProcessor(const std::string& name);

    /**
     * @brief Loads an image from a file.
     * 
     * @param path The absolute or relative path to the image file.
     * @return true If the image was loaded successfully.
     * @return false If the file could not be found or format is invalid.
     */
    bool load(const std::string& path);

    /**
     * @brief Applies a Gaussian blur to the loaded image.
     * 
     * @param kernelSize The size of the kernel. Must be an odd number (e.g., 3, 5, 7).
     * @param sigma The standard deviation of the Gaussian distribution.
     * @throws std::invalid_argument If kernelSize is even or non-positive.
     * @see load()
     */
    void applyBlur(int kernelSize, float sigma);

    /**
     * @brief Gets the current status of the processor.
     * 
     * @return std::string A description of the current state.
     */
    std::string getStatus() const;

private:
    std::string m_name; ///< The name of the processor.
    bool m_isLoaded;    ///< Flag indicating if an image is loaded.
};
