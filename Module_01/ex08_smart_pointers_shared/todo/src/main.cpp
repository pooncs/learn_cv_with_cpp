#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Task 1: The Shared Resource
class Mesh {
public:
    Mesh() { std::cout << "Mesh Loaded (Constructed)\n"; }
    ~Mesh() { std::cout << "Mesh Unloaded (Destructed)\n"; }
};

// Task 3: Node class for Cycle
struct Node {
    std::string name;
    // TODO: One of these needs to be weak_ptr to break the cycle!
    std::shared_ptr<Node> child;
    std::shared_ptr<Node> parent; 

    Node(std::string n) : name(n) { std::cout << "Node " << name << " created\n"; }
    ~Node() { std::cout << "Node " << name << " destroyed\n"; }
};

int main() {
    // ---------------------------------------------------------
    // Task 2: Shared Ownership
    // ---------------------------------------------------------
    {
        std::cout << "--- Shared Ownership ---\n";
        // TODO: Create a shared_ptr to Mesh
        // std::shared_ptr<Mesh> m1 = ...
        
        // TODO: Create a second pointer sharing the same mesh
        // std::shared_ptr<Mesh> m2 = ...
        
        // TODO: Print use_count()
    } // Mesh should be destroyed here

    // ---------------------------------------------------------
    // Task 3: The Cycle Problem
    // ---------------------------------------------------------
    std::cout << "\n--- Cycle Test ---\n";
    {
        auto parent = std::make_shared<Node>("Parent");
        auto child = std::make_shared<Node>("Child");

        parent->child = child;
        child->parent = parent;

        std::cout << "Parent use count: " << parent.use_count() << "\n";
        std::cout << "Child use count: " << child.use_count() << "\n";
        
        // When we leave this scope, destructors should run.
        // IF they don't run, you have a memory leak!
    }

    return 0;
}
