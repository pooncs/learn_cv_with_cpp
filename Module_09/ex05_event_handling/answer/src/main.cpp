#include <QApplication>
#include <QLabel>
#include <QMouseEvent>
#include <iostream>

class ClickableLabel : public QLabel {
public:
    explicit ClickableLabel(QWidget *parent = nullptr) : QLabel(parent) {
        setText("Click me!");
        setAlignment(Qt::AlignCenter);
        setStyleSheet("border: 2px solid black; background-color: lightyellow;");
    }

protected:
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton) {
            std::cout << "Clicked at: " << event->pos().x() << ", " << event->pos().y() << std::endl;
            setText("Clicked: " + QString::number(event->pos().x()) + ", " + QString::number(event->pos().y()));
        }
        // Call base class implementation just in case
        QLabel::mousePressEvent(event);
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    ClickableLabel label;
    label.resize(400, 300);
    label.show();

    return app.exec();
}
