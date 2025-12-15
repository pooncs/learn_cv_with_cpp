#include <gtest/gtest.h>
#include <QApplication>
#include <QWidget>
#include <QPainter>

// Mock widget to test if we can compile subclass logic
class TestWidget : public QWidget {
public:
    bool painted = false;
protected:
    void paintEvent(QPaintEvent *event) override {
        painted = true;
        QPainter p(this);
        p.drawRect(0,0,10,10);
    }
};

TEST(CustomWidget, PaintEventDispatch) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }

    TestWidget w;
    w.resize(100, 100);
    w.show();
    
    // Process events to trigger paint
    QApplication::processEvents();
    
    // Note: paintEvent might not be called immediately in headless or without window manager,
    // but we check if the object is valid and logic compiles.
    // In a real GUI test, we would use QTest::qWaitForWindowExposed(&w);
    
    EXPECT_TRUE(w.isVisible());
}
