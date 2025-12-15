#include <QApplication>
#include <QMainWindow>

int main(int argc, char *argv[]) {
    // 1. Create the application instance
    QApplication app(argc, argv);

    // 2. Create the main window
    QMainWindow window;
    window.setWindowTitle("Hello Qt");
    window.resize(800, 600);

    // 3. Show the window
    window.show();

    // 4. Run the event loop
    return app.exec();
}
