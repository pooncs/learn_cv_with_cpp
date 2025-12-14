#include <iostream>
#include <memory>
#include <vector>

struct Mesh {
    std::string name;
    Mesh(std::string n) : name(n) { std::cout << "Mesh " << name << " created\n"; }
    ~Mesh() { std::cout << "Mesh " << name << " destroyed\n"; }
};

struct Node {
    // TODO: Use shared_ptr for mesh
    // std::shared_ptr<Mesh> mesh;
    std::string id;
};

int main() {
    // TODO: Create a shared mesh
    
    // TODO: Create multiple nodes sharing that mesh
    
    // TODO: Clear nodes and check if mesh persists (it should if you hold a ref, or shouldn't if not)
    
    return 0;
}
