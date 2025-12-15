#include <gtest/gtest.h>
#include <QApplication>
#include <QMainWindow>

TEST(QtSetup, ObjectCreation) {
    // We can't easily run a full QApplication in a unit test without a GUI environment sometimes,
    // but we can test object instantiation if we are careful.
    // However, QApplication requires argc/argv.
    
    int argc = 0;
    char *argv[] = { nullptr };
    
    // Check if we can create a QApplication instance (if one doesn't exist)
    if (!QApplication::instance()) {
        QApplication app(argc, argv);
        QMainWindow window;
        window.setWindowTitle("Test Window");
        EXPECT_EQ(window.windowTitle(), "Test Window");
    } else {
        QMainWindow window;
        window.setWindowTitle("Test Window");
        EXPECT_EQ(window.windowTitle(), "Test Window");
    }
}
