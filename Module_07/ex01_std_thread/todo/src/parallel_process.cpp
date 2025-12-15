#include "parallel_process.hpp"

namespace cv_curriculum {

// Helper function to process a strip
static void invertStrip(cv::Mat& img, int startRow, int endRow) {
    // TODO: Iterate from startRow to endRow and invert pixels
    // Use img.ptr<uchar>(y) for speed
}

void invertImageSequential(cv::Mat& img) {
    invertStrip(img, 0, img.rows);
}

void invertImageParallel(cv::Mat& img, int numThreads) {
    std::vector<std::thread> threads;
    
    // TODO: Implement Parallel Processing
    // 1. Calculate split (rows per thread)
    // 2. Loop i from 0 to numThreads
    // 3. Calculate start/end row
    // 4. threads.emplace_back(invertStrip, std::ref(img), start, end)
    // 5. Join all threads
}

} // namespace cv_curriculum
