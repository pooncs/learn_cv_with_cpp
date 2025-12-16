# Exercise 01: Documentation Management

## Goal
Generate professional API documentation for C++ code using **Doxygen** and for Python bindings using **Sphinx**.

## Learning Objectives
1.  **Doxygen Comments:** Write Javadoc-style comments (`/** ... */`) for classes and functions.
2.  **Doxyfile Configuration:** Configure `Doxyfile` to parse C++ source and generate HTML.
3.  **Graphviz Integration:** Use `HAVE_DOT = YES` to generate class hierarchy and call graphs.
4.  **Sphinx (Optional):** Understand how to generate Python docs (often used for Python bindings of C++ libs).

## Practical Motivation
Code without documentation is a "write-only" memory. In large teams or open-source projects, auto-generated documentation is the standard way to help others understand your API without reading every line of implementation.

**Analogy:**
*   **Source Code:** The blueprint of a building. Detailed, technical, hard to read for a visitor.
*   **Documentation (Doxygen):** The brochure and map of the building. It explains "This is the Lobby," "This is the Exit," without showing the wiring inside the walls.

## Theory: Doxygen Tags
*   `@brief`: Short description.
*   `@param`: Parameter description.
*   `@return`: Return value description.
*   `@note`: Important warnings or details.
*   `@see`: References to other classes/methods.

## Step-by-Step Instructions

### Task 1: Annotate Code
1.  Open `src/my_class.h`.
2.  Add Doxygen comments to the class and its methods.

### Task 2: Generate Doxyfile
1.  Run `doxygen -g` (if installed) or create a file named `Doxyfile`.
2.  Edit `Doxyfile`:
    *   `PROJECT_NAME = "My CV Library"`
    *   `INPUT = src`
    *   `RECURSIVE = YES`
    *   `GENERATE_HTML = YES`

### Task 3: Build Docs
1.  Run `doxygen Doxyfile`.
2.  Open `html/index.html` in a browser.

## Code Hints
```cpp
/**
 * @brief A class representing a simple image processor.
 * 
 * This class provides methods to load, process, and save images.
 */
class ImageProcessor {
public:
    /**
     * @brief Blurs the image.
     * 
     * @param kernelSize The size of the blur kernel (must be odd).
     * @return True if successful, false otherwise.
     * @note This operation modifies the internal image buffer.
     */
    bool blur(int kernelSize);
};
```

## Verification
The `html` folder should be created. Opening `index.html` should show your class documentation with the descriptions you wrote.
