#include <iostream>
#include <memory>
#include <vector>

struct Mesh {
    std::string name;
    Mesh(std::string n) : name(n) { std::cout << "Mesh " << name << " created\n"; }
    ~Mesh() { std::cout << "Mesh " << name << " destroyed\n"; }
};

struct Node {
    std::shared_ptr<Mesh> mesh;
    std::string id;
    
    Node(std::string i, std::shared_ptr<Mesh> m) : id(i), mesh(m) {}
};

int main() {
    auto sphere_mesh = std::make_shared<Mesh>("Sphere");
    
    std::vector<Node> scene;
    scene.emplace_back("Player1", sphere_mesh);
    scene.emplace_back("Player2", sphere_mesh);
    
    std::cout << "Mesh use count: " << sphere_mesh.use_count() << "\n";
    
    scene.clear(); // Nodes destroyed
    
    std::cout << "Mesh use count after clear: " << sphere_mesh.use_count() << "\n";
    // sphere_mesh still exists because of local variable
    
    return 0;
}
