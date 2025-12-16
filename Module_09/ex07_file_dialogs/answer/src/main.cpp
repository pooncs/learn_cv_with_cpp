#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    QVBoxLayout *layout = new QVBoxLayout(&window);

    QPushButton *btnOpen = new QPushButton("Open Image");
    QPushButton *btnSave = new QPushButton("Save Text");
    QLabel *lblPath = new QLabel("No file selected");

    layout->addWidget(btnOpen);
    layout->addWidget(btnSave);
    layout->addWidget(lblPath);

    QObject::connect(btnOpen, &QPushButton::clicked, [&]() {
        QString fileName = QFileDialog::getOpenFileName(&window, "Open Image", "", "Images (*.png *.jpg *.bmp)");
        if (!fileName.isEmpty()) {
            lblPath->setText("Selected: " + fileName);
        }
    });

    QObject::connect(btnSave, &QPushButton::clicked, [&]() {
        QString fileName = QFileDialog::getSaveFileName(&window, "Save Result", "", "Text Files (*.txt)");
        if (!fileName.isEmpty()) {
            QFile file(fileName);
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&file);
                out << "This is a test result file.\n";
                file.close();
                QMessageBox::information(&window, "Success", "File saved successfully.");
            } else {
                QMessageBox::critical(&window, "Error", "Could not save file.");
            }
        }
    });

    window.setWindowTitle("File Dialogs");
    window.resize(400, 200);
    window.show();

    return app.exec();
}
