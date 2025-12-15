#include <gtest/gtest.h>
#include <QApplication>
#include <QSlider>
#include <QLabel>
#include <QSignalSpy>

TEST(SignalsSlots, Connection) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }

    QSlider slider(Qt::Horizontal);
    QLabel label;
    
    // Mimic the logic we want to test: connection
    QObject::connect(&slider, &QSlider::valueChanged, [&label](int value) {
        label.setNum(value);
    });

    // Spy on the signal
    QSignalSpy spy(&slider, &QSlider::valueChanged);

    // Trigger signal
    slider.setValue(42);

    // Check spy
    EXPECT_EQ(spy.count(), 1);
    
    // Check effect
    EXPECT_EQ(label.text().toInt(), 42);
}
