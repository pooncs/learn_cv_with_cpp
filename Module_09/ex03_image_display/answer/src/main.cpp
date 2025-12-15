#include <QApplication>
#include <QLabel>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // 1. Load Image
    // Note: In a real app, use absolute path or configure working directory
    cv::Mat mat = cv::imread("data/lenna.png");
    if (mat.empty()) {
        // Create a dummy image if file not found
        mat = cv::Mat(400, 600, CV_8UC3, cv::Scalar(0, 255, 0));
        cv::putText(mat, "Image Not Found", cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }

    // 2. Convert BGR to RGB
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    // 3. Create QImage
    // data, width, height, bytesPerLine, format
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);

    // 4. Display
    QLabel label;
    label.setPixmap(QPixmap::fromImage(qimg));
    label.setWindowTitle("OpenCV Image in Qt");
    label.resize(mat.cols, mat.rows);
    label.show();

    return app.exec();
}
