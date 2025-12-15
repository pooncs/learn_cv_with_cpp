#include "stitcher.hpp"

namespace cv_curriculum {

cv::Mat stitchImages(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat panorama;
    
    // TODO: Implement Panorama Stitcher
    // 1. Detect ORB features in both images.
    // 2. Match descriptors (BF + Ratio Test).
    // 3. Compute Homography (img2 -> img1).
    // 4. Warp img2 using H.
    //    - Careful with canvas size (warped image might go out of bounds).
    //    - You may need to translate img1 too if warped img2 has negative coordinates.
    // 5. Blend img1 and warped img2.
    
    return panorama;
}

} // namespace cv_curriculum
