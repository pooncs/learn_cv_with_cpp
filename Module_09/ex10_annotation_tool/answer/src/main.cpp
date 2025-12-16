#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QPainter>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>
#include <QVector>
#include <QRect>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class AnnotatorWidget : public QWidget {
public:
    explicit AnnotatorWidget(QWidget *parent = nullptr) : QWidget(parent) {
        setMouseTracking(true);
    }

    void setImage(const QImage &img) {
        m_image = img;
        m_boxes.clear();
        resize(m_image.size());
        update();
    }

    const QVector<QRect>& getBoxes() const { return m_boxes; }

protected:
    void paintEvent(QPaintEvent *event) override {
        QPainter painter(this);
        if (!m_image.isNull()) {
            painter.drawImage(0, 0, m_image);
        }

        // Draw existing boxes
        painter.setPen(QPen(Qt::green, 2));
        for (const auto &box : m_boxes) {
            painter.drawRect(box);
        }

        // Draw current drawing box
        if (m_drawing) {
            painter.setPen(QPen(Qt::red, 2, Qt::DashLine));
            painter.drawRect(QRect(m_startPoint, m_endPoint));
        }
    }

    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton) {
            m_drawing = true;
            m_startPoint = event->pos();
            m_endPoint = m_startPoint;
        }
    }

    void mouseMoveEvent(QMouseEvent *event) override {
        if (m_drawing) {
            m_endPoint = event->pos();
            update();
        }
    }

    void mouseReleaseEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton && m_drawing) {
            m_drawing = false;
            m_endPoint = event->pos();
            m_boxes.append(QRect(m_startPoint, m_endPoint).normalized());
            update();
        }
    }

private:
    QImage m_image;
    QVector<QRect> m_boxes;
    QPoint m_startPoint, m_endPoint;
    bool m_drawing = false;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    QVBoxLayout *mainLayout = new QVBoxLayout(&window);

    // Toolbar
    QHBoxLayout *toolbar = new QHBoxLayout();
    QPushButton *btnOpen = new QPushButton("Open Image");
    QPushButton *btnSave = new QPushButton("Save JSON");
    toolbar->addWidget(btnOpen);
    toolbar->addWidget(btnSave);

    // Annotator View
    AnnotatorWidget *annotator = new AnnotatorWidget();
    
    // Default image
    QImage dummy(800, 600, QImage::Format_RGB888);
    dummy.fill(Qt::black);
    annotator->setImage(dummy);

    mainLayout->addLayout(toolbar);
    mainLayout->addWidget(annotator);

    QObject::connect(btnOpen, &QPushButton::clicked, [&]() {
        QString fileName = QFileDialog::getOpenFileName(&window, "Open Image", "", "Images (*.png *.jpg)");
        if (!fileName.isEmpty()) {
            QImage img(fileName);
            if (!img.isNull()) {
                annotator->setImage(img);
            }
        }
    });

    QObject::connect(btnSave, &QPushButton::clicked, [&]() {
        QString fileName = QFileDialog::getSaveFileName(&window, "Save Labels", "", "JSON (*.json)");
        if (!fileName.isEmpty()) {
            json j_boxes = json::array();
            for (const auto &box : annotator->getBoxes()) {
                j_boxes.push_back({
                    {"x", box.x()},
                    {"y", box.y()},
                    {"w", box.width()},
                    {"h", box.height()}
                });
            }
            std::ofstream o(fileName.toStdString());
            o << j_boxes.dump(4) << std::endl;
            QMessageBox::information(&window, "Saved", "Annotations saved to JSON.");
        }
    });

    window.setWindowTitle("Simple Annotator");
    window.resize(850, 700);
    window.show();

    return app.exec();
}
