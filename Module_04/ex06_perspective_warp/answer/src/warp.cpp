#include "warp.hpp"
#include <numeric>
#include <algorithm>

std::vector<cv::Point2f> sort_corners(const std::vector<cv::Point2f>& pts) {
    CV_Assert(pts.size() == 4);
    std::vector<cv::Point2f> sorted(4);
    
    // Sort by Y first
    std::vector<cv::Point2f> sorted_y = pts;
    std::sort(sorted_y.begin(), sorted_y.end(), [](const cv::Point2f& a, const cv::Point2f& b){
        return a.y < b.y;
    });

    // Top 2 are TL, TR. Sort by X
    if (sorted_y[0].x < sorted_y[1].x) {
        sorted[0] = sorted_y[0]; // TL
        sorted[1] = sorted_y[1]; // TR
    } else {
        sorted[0] = sorted_y[1];
        sorted[1] = sorted_y[0];
    }

    // Bottom 2 are BR, BL. Sort by X
    if (sorted_y[2].x < sorted_y[3].x) {
        sorted[3] = sorted_y[2]; // BL
        sorted[2] = sorted_y[3]; // BR
    } else {
        sorted[3] = sorted_y[3];
        sorted[2] = sorted_y[2];
    }
    
    return sorted;
}

cv::Mat rectify_document(const cv::Mat& src, const std::vector<cv::Point2f>& corners, float aspect_ratio) {
    auto sorted_pts = sort_corners(corners);
    
    // Estimate width: max(dist(TL, TR), dist(BL, BR))
    float w1 = cv::norm(sorted_pts[0] - sorted_pts[1]);
    float w2 = cv::norm(sorted_pts[3] - sorted_pts[2]);
    float maxWidth = std::max(w1, w2);

    // Estimate height: max(dist(TL, BL), dist(TR, BR))
    float h1 = cv::norm(sorted_pts[0] - sorted_pts[3]);
    float h2 = cv::norm(sorted_pts[1] - sorted_pts[2]);
    float maxHeight = std::max(h1, h2);

    // Or enforce aspect ratio
    if (aspect_ratio > 0) {
        // Assume width is primary? Or fit to bounding box?
        // Let's use maxWidth and compute height from AR
        // AR = W / H -> H = W / AR
        maxHeight = maxWidth / aspect_ratio;
    }

    std::vector<cv::Point2f> dst_pts = {
        {0, 0},
        {maxWidth - 1, 0},
        {maxWidth - 1, maxHeight - 1},
        {0, maxHeight - 1}
    };

    cv::Mat H = cv::getPerspectiveTransform(sorted_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(src, warped, H, cv::Size((int)maxWidth, (int)maxHeight));
    
    return warped;
}
