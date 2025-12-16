#include <iostream>
#include <cstdio>
#include <utility> // for std::exchange

class FileHandle {
    FILE* file = nullptr;

public:
    // Constructor
    FileHandle(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (file) {
            std::cout << "FileHandle: Opened " << filename << "\n";
        } else {
            std::cout << "FileHandle: Failed to open " << filename << "\n";
        }
    }

    // Destructor
    ~FileHandle() {
        if (file) {
            fclose(file);
            std::cout << "FileHandle: Closed file\n";
        }
    }

    // Accessor
    FILE* get() const { return file; }

    // Rule of Five: Delete Copy
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // Rule of Five: Implement Move Constructor
    FileHandle(FileHandle&& other) noexcept 
        : file(std::exchange(other.file, nullptr)) {
        std::cout << "FileHandle: Move Constructor\n";
    }

    // Rule of Five: Implement Move Assignment
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            // Close current file if any
            if (file) fclose(file);
            
            // Steal resources
            file = std::exchange(other.file, nullptr);
            std::cout << "FileHandle: Move Assignment\n";
        }
        return *this;
    }
};

void write_log(FileHandle fh) {
    if (fh.get()) {
        fprintf(fh.get(), "Log entry: System is running.\n");
        std::cout << "Writing to log inside function...\n";
    }
} // fh is destroyed here, file closed

int main() {
    std::cout << "--- Start ---\n";

    FileHandle log("log.txt", "w");
    
    // FileHandle copy = log; // ERROR: Call to deleted constructor

    std::cout << "Moving log to function...\n";
    write_log(std::move(log)); 
    
    if (log.get() == nullptr) {
        std::cout << "Main: 'log' variable is now empty.\n";
    }

    std::cout << "--- End of Main ---\n";
    return 0;
}
