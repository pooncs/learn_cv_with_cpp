#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

// Simple Box structure
struct Box {
    float x, y, w, h;
};

class KalmanTracker {
public:
    int id;
    int time_since_update;
    int hits;
    int hit_streak;
    int age;
    cv::KalmanFilter kf;
    cv::Mat measurement;

    static int count;

    KalmanTracker(Box initBox) {
        id = count++;
        time_since_update = 0;
        hits = 0;
        hit_streak = 0;
        age = 0;

        // State: [x, y, area, ratio, vx, vy, v_area]
        // Note: For simplicity, we'll use [x, y, w, h, vx, vy, vw, vh] (Constant Velocity)
        kf.init(8, 4, 0);
        kf.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
        
        // dt = 1
        kf.transitionMatrix.at<float>(0, 4) = 1;
        kf.transitionMatrix.at<float>(1, 5) = 1;
        kf.transitionMatrix.at<float>(2, 6) = 1;
        kf.transitionMatrix.at<float>(3, 7) = 1;

        kf.measurementMatrix = cv::Mat::eye(4, 8, CV_32F);
        
        // Initialize state
        kf.statePost.at<float>(0) = initBox.x;
        kf.statePost.at<float>(1) = initBox.y;
        kf.statePost.at<float>(2) = initBox.w;
        kf.statePost.at<float>(3) = initBox.h;
        
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

        measurement = cv::Mat::zeros(4, 1, CV_32F);
    }

    Box predict() {
        cv::Mat p = kf.predict();
        age += 1;
        if (time_since_update > 0) hit_streak = 0;
        time_since_update += 1;
        return {p.at<float>(0), p.at<float>(1), p.at<float>(2), p.at<float>(3)};
    }

    void update(Box stateMat) {
        time_since_update = 0;
        hits += 1;
        hit_streak += 1;

        measurement.at<float>(0) = stateMat.x;
        measurement.at<float>(1) = stateMat.y;
        measurement.at<float>(2) = stateMat.w;
        measurement.at<float>(3) = stateMat.h;

        kf.correct(measurement);
    }

    Box getState() {
        return {kf.statePost.at<float>(0), kf.statePost.at<float>(1), 
                kf.statePost.at<float>(2), kf.statePost.at<float>(3)};
    }
};

int KalmanTracker::count = 0;

// Simple IoU
float iou(Box a, Box b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    if (x2 < x1 || y2 < y1) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float areaA = a.w * a.h;
    float areaB = b.w * b.h;
    
    return intersection / (areaA + areaB - intersection);
}

int main() {
    std::vector<KalmanTracker> trackers;
    int max_age = 5;
    int min_hits = 3;
    float iouThreshold = 0.3f;

    // Simulated Detections for 10 frames
    // Object moving diagonally from (0,0) -> (100,100)
    for (int frame = 0; frame < 10; ++frame) {
        std::cout << "--- Frame " << frame << " ---" << std::endl;
        
        std::vector<Box> detections;
        // Detection noise
        float noise = (rand() % 10) / 10.0f;
        detections.push_back({(float)frame * 10 + noise, (float)frame * 10 + noise, 20, 20});

        // 1. Predict Tracks
        for (auto& trk : trackers) {
            Box p = trk.predict();
            std::cout << "Track " << trk.id << " predicted at: " << p.x << ", " << p.y << std::endl;
        }

        // 2. Associate (Greedy for simplicity, usually Hungarian)
        std::vector<int> assignment(detections.size(), -1);
        std::vector<bool> track_assigned(trackers.size(), false);

        for (int i = 0; i < detections.size(); ++i) {
            float best_iou = 0;
            int best_track = -1;
            
            for (int t = 0; t < trackers.size(); ++t) {
                if (track_assigned[t]) continue;
                
                float sim = iou(detections[i], trackers[t].getState());
                if (sim > best_iou) {
                    best_iou = sim;
                    best_track = t;
                }
            }

            if (best_iou > iouThreshold) {
                assignment[i] = best_track;
                track_assigned[best_track] = true;
            }
        }

        // 3. Update Matched
        for (int i = 0; i < detections.size(); ++i) {
            if (assignment[i] != -1) {
                std::cout << "Matched Det " << i << " to Track " << trackers[assignment[i]].id << std::endl;
                trackers[assignment[i]].update(detections[i]);
            } else {
                std::cout << "New Track created for Det " << i << std::endl;
                trackers.push_back(KalmanTracker(detections[i]));
            }
        }

        // 4. Delete Dead Tracks
        for (auto it = trackers.begin(); it != trackers.end();) {
            if (it->time_since_update > max_age) {
                std::cout << "Track " << it->id << " deleted." << std::endl;
                it = trackers.erase(it);
            } else {
                ++it;
            }
        }
    }

    return 0;
}
