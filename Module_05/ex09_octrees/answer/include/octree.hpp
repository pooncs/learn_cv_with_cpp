#pragma once
#include <vector>
#include <memory>
#include <array>

struct Point3D {
    float x, y, z;
};

struct AABB {
    Point3D min, max;
    bool contains(const Point3D& p) const;
    bool intersects_sphere(const Point3D& center, float radius) const;
};

class Octree {
public:
    Octree(const AABB& boundary, int capacity = 4);
    
    bool insert(const Point3D& p);
    std::vector<Point3D> query_radius(const Point3D& center, float radius) const;

private:
    struct Node {
        AABB boundary;
        std::vector<Point3D> points;
        std::array<std::unique_ptr<Node>, 8> children;
        bool is_leaf = true;
        
        Node(const AABB& b) : boundary(b) {}
    };
    
    std::unique_ptr<Node> root;
    int capacity;
    
    void subdivide(Node* node);
    bool insert_recursive(Node* node, const Point3D& p);
    void query_recursive(Node* node, const Point3D& center, float radius, std::vector<Point3D>& results) const;
};
