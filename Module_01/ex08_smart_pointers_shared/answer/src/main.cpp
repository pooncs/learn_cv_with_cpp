#include <iostream>
#include <memory>
#include <string>

// Task 1: The Shared Resource
class Mesh {
public:
    Mesh() { std::cout << "Mesh Loaded (Constructed)\n"; }
    ~Mesh() { std::cout << "Mesh Unloaded (Destructed)\n"; }
};

struct Node {
    std::string name;
    std::shared_ptr<Node> child;
    // We use weak_ptr for the parent back-pointer to avoid a reference cycle.
    // If this were shared_ptr, Parent keeps Child alive, Child keeps Parent alive -> Leak.
    std::weak_ptr<Node> parent; 

    Node(std::string n) : name(n) { std::cout << "Node " << name << " created\n"; }
    ~Node() { std::cout << "Node " << name << " destroyed\n"; }
};

int main() {
    // ---------------------------------------------------------
    // Task 2: Shared Ownership
    // ---------------------------------------------------------
    {
        std::cout << "--- Shared Ownership ---\n";
        auto m1 = std::make_shared<Mesh>();
        std::cout << "Count after m1: " << m1.use_count() << "\n"; // 1
        
        {
            auto m2 = m1; // Share ownership
            std::cout << "Count after m2: " << m1.use_count() << "\n"; // 2
            // Both m1 and m2 point to the same object
        } // m2 goes out of scope, count decrements to 1. Mesh is NOT destroyed.
        
        std::cout << "Count after m2 scope: " << m1.use_count() << "\n"; // 1
    } // m1 goes out of scope, count -> 0. Mesh destroyed.

    // ---------------------------------------------------------
    // Task 3: The Cycle Problem
    // ---------------------------------------------------------
    std::cout << "\n--- Cycle Test ---\n";
    {
        auto parent = std::make_shared<Node>("Parent");
        auto child = std::make_shared<Node>("Child");

        parent->child = child;
        child->parent = parent; // Assigns shared_ptr to weak_ptr

        std::cout << "Parent use count: " << parent.use_count() << "\n"; // 1 (Child's weak_ptr doesn't count)
        std::cout << "Child use count: " << child.use_count() << "\n"; // 2 (Main + Parent)
        
    } // Both destroyed properly because cycle is broken.

    return 0;
}
