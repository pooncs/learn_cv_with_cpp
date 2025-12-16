#include <iostream>
#include <cstdio> // for FILE, fopen, fclose, fprintf

// Task 1: RAII Wrapper Class
class FileHandle {
    FILE* file = nullptr;

public:
    // Constructor: Opens the file
    FileHandle(const char* filename, const char* mode) {
        // TODO: Implement fopen and check for errors
        std::cout << "FileHandle: Opening " << filename << "\n";
    }

    // Destructor: Closes the file
    ~FileHandle() {
        // TODO: Implement fclose if file is valid
        std::cout << "FileHandle: Closing file\n";
    }

    // Accessor for usage
    FILE* get() const { return file; }

    // Task 2: Delete Copy Operations
    // TODO: Delete Copy Ctor and Copy Assign

    // Task 3: Implement Move Operations
    // TODO: Implement Move Ctor
    // TODO: Implement Move Assign
};

void write_log(FileHandle fh) {
    if (fh.get()) {
        fprintf(fh.get(), "Log entry: System is running.\n");
        std::cout << "Writing to log...\n";
    }
}

int main() {
    std::cout << "--- Start ---\n";

    // Task 4: Usage
    // FileHandle log("log.txt", "w");
    
    // write_log(std::move(log)); // Should invoke move constructor

    std::cout << "--- End of Main ---\n";
    return 0;
}
