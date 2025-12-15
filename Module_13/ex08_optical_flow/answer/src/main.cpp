#include "optical_flow.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv_tracking;

// Helper to draw points
void drawPoints(Mat& img, const vector<Point2f>& points, const Scalar& color) {
    for (const auto& pt : points) {
        circle(img, pt, 3, color, -1);
    }
}

int main() {
    cout << "Module 13 Ex 08: Optical Flow (Lucas-Kanade)" << endl;
    cout << "============================================" << endl;

    // 1. Generate Synthetic Data
    int width = 400;
    int height = 400;
    Mat img1 = Mat::zeros(height, width, CV_8UC1);
    Mat img2 = Mat::zeros(height, width, CV_8UC1);

    // Create a moving pattern (a textured square is better for gradients than a solid one)
    // Solid square has zero gradient inside, only at edges. 
    // Let's create a noisy patch or a simple pattern.
    Rect box1(100, 100, 50, 50);
    Rect box2(105, 103, 50, 50); // Moved by (5, 3)

    // Fill with random noise to ensure gradients exist everywhere inside the box
    RNG rng(12345);
    Mat noise(height, width, CV_8UC1);
    rng.fill(noise, RNG::UNIFORM, 0, 255);

    // Apply noise only within the boxes
    Mat boxRegion1 = img1(box1);
    noise(box1).copyTo(boxRegion1);
    
    Mat boxRegion2 = img2(box2);
    noise(box1).copyTo(boxRegion2); // Copy the SAME noise pattern to the new location

    // Also add some background texture to avoid "aperture problem" everywhere else? 
    // Actually, we only track points inside the box.
    
    // Select points to track (features)
    // Let's pick some points inside the first box
    vector<Point2f> prevPts;
    prevPts.push_back(Point2f(125, 125)); // Center
    prevPts.push_back(Point2f(110, 110)); // Top-left area
    prevPts.push_back(Point2f(140, 140)); // Bottom-right area
    prevPts.push_back(Point2f(125, 110)); // Top-mid
    
    // 2. Run Custom Optical Flow
    cout << "\nRunning Custom Optical Flow..." << endl;
    OpticalFlowTracker tracker;
    vector<Point2f> nextPtsCustom;
    vector<uchar> statusCustom;
    
    tracker.computeFlowCustom(img1, img2, prevPts, nextPtsCustom, statusCustom);

    // 3. Run OpenCV Optical Flow
    cout << "\nRunning OpenCV Optical Flow..." << endl;
    vector<Point2f> nextPtsCV;
    vector<uchar> statusCV;
    tracker.computeFlowOpenCV(img1, img2, prevPts, nextPtsCV, statusCV);

    // 4. Compare Results
    cout << "\nResults Comparison (Expected Shift: dx=5, dy=3):" << endl;
    cout << "------------------------------------------------" << endl;
    
    float totalErrorCustom = 0;
    float totalErrorCV = 0;
    int validPoints = 0;

    for (size_t i = 0; i < prevPts.size(); ++i) {
        cout << "Point " << i << ": (" << prevPts[i].x << ", " << prevPts[i].y << ")" << endl;
        
        if (statusCustom[i]) {
            Point2f diff = nextPtsCustom[i] - prevPts[i];
            cout << "  Custom: (" << nextPtsCustom[i].x << ", " << nextPtsCustom[i].y 
                 << ") Shift: (" << diff.x << ", " << diff.y << ")" << endl;
            totalErrorCustom += sqrt(pow(diff.x - 5.0f, 2) + pow(diff.y - 3.0f, 2));
        } else {
            cout << "  Custom: Lost" << endl;
        }

        if (statusCV[i]) {
            Point2f diff = nextPtsCV[i] - prevPts[i];
            cout << "  OpenCV: (" << nextPtsCV[i].x << ", " << nextPtsCV[i].y 
                 << ") Shift: (" << diff.x << ", " << diff.y << ")" << endl;
            totalErrorCV += sqrt(pow(diff.x - 5.0f, 2) + pow(diff.y - 3.0f, 2));
        } else {
            cout << "  OpenCV: Lost" << endl;
        }
        
        if (statusCustom[i] && statusCV[i]) validPoints++;
    }

    if (validPoints > 0) {
        cout << "\nAverage Error Custom: " << totalErrorCustom / validPoints << " px" << endl;
        cout << "Average Error OpenCV: " << totalErrorCV / validPoints << " px" << endl;
    }

    // Save visualization
    Mat vis;
    cvtColor(img1, vis, COLOR_GRAY2BGR);
    for (size_t i = 0; i < prevPts.size(); ++i) {
        circle(vis, prevPts[i], 3, Scalar(0, 0, 255), -1); // Red: Original
        if (statusCustom[i]) {
            line(vis, prevPts[i], nextPtsCustom[i], Scalar(0, 255, 0), 2); // Green: Custom flow
            circle(vis, nextPtsCustom[i], 3, Scalar(0, 255, 0), -1);
        }
        if (statusCV[i]) {
             circle(vis, nextPtsCV[i], 2, Scalar(255, 0, 0), -1); // Blue: OpenCV
        }
    }
    imwrite("optical_flow_result.png", vis);
    cout << "\nVisualization saved to 'optical_flow_result.png'" << endl;

    return 0;
}
