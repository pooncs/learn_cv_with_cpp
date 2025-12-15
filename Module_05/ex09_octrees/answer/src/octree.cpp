#include "octree.hpp"
#include <cmath>
#include <algorithm>

bool AABB::contains(const Point3D& p) const {
    return p.x >= min.x && p.x <= max.x &&
           p.y >= min.y && p.y <= max.y &&
           p.z >= min.z && p.z <= max.z;
}

bool AABB::intersects_sphere(const Point3D& center, float radius) const {
    float d_min = 0;
    
    if (center.x < min.x) d_min += (center.x - min.x) * (center.x - min.x);
    else if (center.x > max.x) d_min += (center.x - max.x) * (center.x - max.x);
    
    if (center.y < min.y) d_min += (center.y - min.y) * (center.y - min.y);
    else if (center.y > max.y) d_min += (center.y - max.y) * (center.y - max.y);
    
    if (center.z < min.z) d_min += (center.z - min.z) * (center.z - min.z);
    else if (center.z > max.z) d_min += (center.z - max.z) * (center.z - max.z);
    
    return d_min <= radius * radius;
}

Octree::Octree(const AABB& boundary, int capacity) : capacity(capacity) {
    root = std::make_unique<Node>(boundary);
}

bool Octree::insert(const Point3D& p) {
    return insert_recursive(root.get(), p);
}

void Octree::subdivide(Node* node) {
    float midX = (node->boundary.min.x + node->boundary.max.x) / 2;
    float midY = (node->boundary.min.y + node->boundary.max.y) / 2;
    float midZ = (node->boundary.min.z + node->boundary.max.z) / 2;
    
    // Create 8 children
    // 0: ---, 1: --+, 2: -+-, 3: -++, 4: +--, 5: +-+, 6: ++-, 7: +++
    for(int i=0; i<8; ++i) {
        Point3D newMin = node->boundary.min;
        Point3D newMax = node->boundary.max;
        
        if (i & 4) newMin.x = midX; else newMax.x = midX;
        if (i & 2) newMin.y = midY; else newMax.y = midY;
        if (i & 1) newMin.z = midZ; else newMax.z = midZ;
        
        node->children[i] = std::make_unique<Node>(AABB{newMin, newMax});
    }
    node->is_leaf = false;
    
    // Re-distribute existing points
    for(const auto& p : node->points) {
        for(int i=0; i<8; ++i) {
            if(node->children[i]->boundary.contains(p)) {
                insert_recursive(node->children[i].get(), p);
                break;
            }
        }
    }
    node->points.clear();
}

bool Octree::insert_recursive(Node* node, const Point3D& p) {
    if (!node->boundary.contains(p)) return false;
    
    if (node->is_leaf) {
        if (node->points.size() < (size_t)capacity) {
            node->points.push_back(p);
            return true;
        }
        subdivide(node);
        // Fall through to insert into children
    }
    
    for(int i=0; i<8; ++i) {
        if(node->children[i]->boundary.contains(p)) {
            return insert_recursive(node->children[i].get(), p);
        }
    }
    return false; // Should not reach here if boundary contains p
}

std::vector<Point3D> Octree::query_radius(const Point3D& center, float radius) const {
    std::vector<Point3D> results;
    query_recursive(root.get(), center, radius, results);
    return results;
}

void Octree::query_recursive(Node* node, const Point3D& center, float radius, std::vector<Point3D>& results) const {
    if (!node->boundary.intersects_sphere(center, radius)) return;
    
    if (node->is_leaf) {
        for(const auto& p : node->points) {
            float dist_sq = (p.x-center.x)*(p.x-center.x) + (p.y-center.y)*(p.y-center.y) + (p.z-center.z)*(p.z-center.z);
            if (dist_sq <= radius * radius) {
                results.push_back(p);
            }
        }
    } else {
        for(int i=0; i<8; ++i) {
            query_recursive(node->children[i].get(), center, radius, results);
        }
    }
}
