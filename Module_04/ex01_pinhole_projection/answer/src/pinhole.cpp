#include "pinhole.hpp"

cv::Mat project_points(const cv::Mat& points_3d, const cv::Mat& K) {
    CV_Assert(points_3d.cols == 3);
    CV_Assert(K.rows == 3 && K.cols == 3);

    int N = points_3d.rows;
    cv::Mat points_2d(N, 2, points_3d.type());

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int i = 0; i < N; ++i) {
        double X, Y, Z;
        if (points_3d.type() == CV_32F) {
            X = points_3d.at<float>(i, 0);
            Y = points_3d.at<float>(i, 1);
            Z = points_3d.at<float>(i, 2);
        } else {
            X = points_3d.at<double>(i, 0);
            Y = points_3d.at<double>(i, 1);
            Z = points_3d.at<double>(i, 2);
        }

        if (std::abs(Z) < 1e-6) {
            // Handle Z=0 case (e.g., map to infinity or specific value)
            if (points_2d.type() == CV_32F) {
                points_2d.at<float>(i, 0) = -1.0f;
                points_2d.at<float>(i, 1) = -1.0f;
            } else {
                points_2d.at<double>(i, 0) = -1.0;
                points_2d.at<double>(i, 1) = -1.0;
            }
            continue;
        }

        double u = fx * (X / Z) + cx;
        double v = fy * (Y / Z) + cy;

        if (points_2d.type() == CV_32F) {
            points_2d.at<float>(i, 0) = static_cast<float>(u);
            points_2d.at<float>(i, 1) = static_cast<float>(v);
        } else {
            points_2d.at<double>(i, 0) = u;
            points_2d.at<double>(i, 1) = v;
        }
    }

    return points_2d;
}
