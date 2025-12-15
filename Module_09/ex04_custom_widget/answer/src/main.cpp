#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QImage>

class ImageWidget : public QWidget {
public:
    explicit ImageWidget(QWidget *parent = nullptr) : QWidget(parent) {}

    void setImage(const QImage &img) {
        m_image = img;
        update(); // Schedule repaint
        resize(m_image.size());
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        QPainter painter(this);

        if (!m_image.isNull()) {
            painter.drawImage(0, 0, m_image);
        }

        // Draw overlay
        QPen pen(Qt::red);
        pen.setWidth(3);
        painter.setPen(pen);
        painter.drawRect(50, 50, 100, 100);

        painter.setPen(Qt::blue);
        painter.setFont(QFont("Arial", 16));
        painter.drawText(50, 40, "Detected Object");
    }

private:
    QImage m_image;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    ImageWidget widget;
    
    // Create a dummy image
    QImage img(400, 300, QImage::Format_RGB888);
    img.fill(Qt::lightGray);

    widget.setImage(img);
    widget.setWindowTitle("Custom Painting");
    widget.show();

    return app.exec();
}
