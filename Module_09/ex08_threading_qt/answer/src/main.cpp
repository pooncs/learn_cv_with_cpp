#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QProgressBar>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QThread>
#include <iostream>

void heavyTask() {
    // Simulate work
    for (int i = 0; i < 5; ++i) {
        QThread::sleep(1);
        std::cout << "Working... " << i << std::endl;
    }
}

class WorkerWidget : public QWidget {
public:
    WorkerWidget(QWidget *parent = nullptr) : QWidget(parent) {
        QVBoxLayout *layout = new QVBoxLayout(this);
        
        btnStart = new QPushButton("Start Heavy Task");
        progressBar = new QProgressBar();
        progressBar->setRange(0, 0); // Indeterminate mode
        progressBar->hide();

        layout->addWidget(btnStart);
        layout->addWidget(progressBar);

        connect(btnStart, &QPushButton::clicked, this, &WorkerWidget::startTask);
        connect(&watcher, &QFutureWatcher<void>::finished, this, &WorkerWidget::onFinished);
    }

private slots:
    void startTask() {
        btnStart->setEnabled(false);
        progressBar->show();
        
        // Start thread
        QFuture<void> future = QtConcurrent::run(heavyTask);
        watcher.setFuture(future);
    }

    void onFinished() {
        btnStart->setEnabled(true);
        progressBar->hide();
        std::cout << "Task Finished!" << std::endl;
    }

private:
    QPushButton *btnStart;
    QProgressBar *progressBar;
    QFutureWatcher<void> watcher;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    WorkerWidget w;
    w.resize(300, 150);
    w.show();

    return app.exec();
}
