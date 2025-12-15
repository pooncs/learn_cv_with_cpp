#include <iostream>
#include "octree.hpp"

int main() {
    AABB bounds{{0, 0, 0}, {100, 100, 100}};
    Octree tree(bounds, 4);
    
    // TODO: Insert points
    // TODO: Query radius

    return 0;
}
