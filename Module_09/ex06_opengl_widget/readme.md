# Exercise 06: OpenGL Widget

## Goal
Render a simple 3D triangle using `QOpenGLWidget`.

## Learning Objectives
1.  Setup `QOpenGLWidget` and override its lifecycle methods.
2.  Initialize OpenGL functions (`initializeOpenGLFunctions`).
3.  Basic fixed-function or shader-based rendering (we'll stick to basic fixed-function or simple shader for simplicity if possible, but modern Qt recommends shaders).

## Practical Motivation
For high-performance 3D visualization (point clouds, meshes), standard 2D widgets are too slow. `QOpenGLWidget` allows embedding a hardware-accelerated GL context directly into the Qt UI.

## Theory: QOpenGLWidget Lifecycle
1.  `initializeGL()`: Called once. Setup resources (buffers, shaders).
2.  `resizeGL(w, h)`: Called on resize. Setup viewport/projection.
3.  `paintGL()`: Called to render the frame.

## Step-by-Step Instructions

### Task 1: Subclass QOpenGLWidget
Open `todo/src/main.cpp`.
1.  Inherit from `QOpenGLWidget` and `QOpenGLFunctions`.
2.  Override the 3 GL methods.

### Task 2: Initialize
1.  In `initializeGL()`: call `initializeOpenGLFunctions()`.
2.  Set clear color: `glClearColor(...)`.

### Task 3: Render
1.  In `paintGL()`:
    *   `glClear(GL_COLOR_BUFFER_BIT)`.
    *   Draw a triangle using legacy `glBegin(GL_TRIANGLES)` / `glEnd()` for simplicity, or modern VBOs if you are comfortable. (For this intro, we might stick to basic GL commands provided by `QOpenGLFunctions` or just clear screen to prove context works).
    *   *Note:* `QOpenGLFunctions` provides GLES2.0 compatible subset. Legacy `glBegin` might not be available without `QOpenGLFunctions_1_0`. Let's just clear the screen to a specific color to verify context is working, or draw a simple triangle if functions allow.

## Verification
Run the app. You should see a window with the background color you set (e.g., cornflower blue).
