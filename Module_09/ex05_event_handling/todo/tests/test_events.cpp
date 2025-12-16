#include <gtest/gtest.h>
#include <QApplication>
#include <QLabel>
#include <QMouseEvent>

// Mock widget
class TestClickableLabel : public QLabel {
public:
    QPoint lastClick;
    bool clicked = false;

protected:
    void mousePressEvent(QMouseEvent *event) override {
        clicked = true;
        lastClick = event->pos();
    }
};

TEST(EventHandling, MouseClick) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }

    TestClickableLabel label;
    label.resize(100, 100);
    
    // Simulate Event
    QMouseEvent event(QEvent::MouseButtonPress, QPoint(50, 50), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&label, &event);
    
    EXPECT_TRUE(label.clicked);
    EXPECT_EQ(label.lastClick.x(), 50);
    EXPECT_EQ(label.lastClick.y(), 50);
}
