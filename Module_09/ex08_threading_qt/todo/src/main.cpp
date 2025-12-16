#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QProgressBar>
#include <QtConcurrent>
#include <QFutureWatcher>

// TODO: Define heavyTask function

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // TODO: Setup UI
    
    // TODO: Connect button to run heavyTask via QtConcurrent::run
    
    // TODO: Use QFutureWatcher to detect finish
    
    return app.exec();
}
