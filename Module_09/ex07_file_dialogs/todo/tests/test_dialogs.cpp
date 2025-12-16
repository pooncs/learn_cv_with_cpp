#include <gtest/gtest.h>
#include <QApplication>
#include <QFileDialog>
#include <QTimer>

TEST(FileDialogs, OpenDialog) {
    int argc = 0;
    if (!QApplication::instance()) {
        new QApplication(argc, nullptr);
    }
    
    // We cannot interact with native dialogs in automated tests easily.
    // However, we can test QFileDialog class existence and static methods signature compilation.
    
    // Strategy: Use a QTimer to close the dialog immediately after it opens, 
    // or skip the actual exec() call and just verify we can instantiate it.
    
    QFileDialog dialog;
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilter("Images (*.png *.jpg)");
    
    EXPECT_EQ(dialog.fileMode(), QFileDialog::ExistingFile);
    
    // Note: Calling .exec() or static getOpenFileName would block the test until user interaction,
    // so we avoid it here.
}
