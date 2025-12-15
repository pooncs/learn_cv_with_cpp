#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QSlider>
#include <QLabel>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    QVBoxLayout *layout = new QVBoxLayout(&window);

    QSlider *slider = new QSlider(Qt::Horizontal);
    QLabel *label = new QLabel("0");

    layout->addWidget(slider);
    layout->addWidget(label);

    QObject::connect(slider, &QSlider::valueChanged, [label](int value) {
        label->setNum(value);
    });

    window.setWindowTitle("Signals & Slots");
    window.resize(300, 100);
    window.show();

    return app.exec();
}
