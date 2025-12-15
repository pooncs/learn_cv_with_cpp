#include "parallel_process.hpp"

namespace cv_curriculum {

// Worker function for a strip
static void invertStrip(cv::Mat& img, int startRow, int endRow) {
    for (int y = startRow; y < endRow; ++y) {
        uchar* ptr = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols * img.channels(); ++x) {
            ptr[x] = 255 - ptr[x];
        }
    }
}

void invertImageSequential(cv::Mat& img) {
    invertStrip(img, 0, img.rows);
}

void invertImageParallel(cv::Mat& img, int numThreads) {
    std::vector<std::thread> threads;
    int rowsPerThread = img.rows / numThreads;
    
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? img.rows : (i + 1) * rowsPerThread;
        
        // Launch thread
        // Note: passing img by reference requires std::ref or passing pointer/wrapper
        // Since cv::Mat is a reference-counted handle, passing by value is fine (shallow copy)
        // BUT if we modify data, we need to make sure we access the same data. 
        // cv::Mat copy shares data, so it's fine.
        threads.emplace_back(invertStrip, std::ref(img), startRow, endRow);
    }
    
    // Join
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

} // namespace cv_curriculum
