#include "octree.hpp"

bool AABB::contains(const Point3D& p) const {
    // TODO: Check bounds
    return false;
}

bool AABB::intersects_sphere(const Point3D& center, float radius) const {
    // TODO: AABB-Sphere intersection test
    return false;
}

Octree::Octree(const AABB& boundary, int capacity) : capacity(capacity) {
    root = std::make_unique<Node>(boundary);
}

bool Octree::insert(const Point3D& p) {
    // TODO: Recursive insert
    // If leaf and full -> subdivide
    return false;
}

std::vector<Point3D> Octree::query_radius(const Point3D& center, float radius) const {
    // TODO: Recursive query
    return {};
}
