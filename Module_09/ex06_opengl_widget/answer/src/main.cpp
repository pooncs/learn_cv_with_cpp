#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <iostream>

class MyGLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    explicit MyGLWidget(QWidget *parent = nullptr) : QOpenGLWidget(parent) {}

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0.39f, 0.58f, 0.93f, 1.0f); // Cornflower Blue
    }

    void resizeGL(int w, int h) override {
        // Optional: Update projection matrix here
        glViewport(0, 0, w, h);
    }

    void paintGL() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Simple Fixed-Function Triangle (Legacy GL, available in Compatibility profile or some drivers)
        // Note: For modern Core Profile, VBOs/VAOs are required. 
        // QOpenGLFunctions usually gives GLES2.0 which doesn't support immediate mode (glBegin/glEnd).
        // So we will just stick to clearing the screen to prove context works.
        // If we want to draw, we would need to compile shaders.
        // For this exercise, verifying Clear Color is enough to prove GL Context.
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MyGLWidget widget;
    widget.resize(800, 600);
    widget.setWindowTitle("Qt OpenGL Widget");
    widget.show();

    return app.exec();
}
