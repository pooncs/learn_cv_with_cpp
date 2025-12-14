#include <iostream>
#include <variant>
#include <vector>

struct Circle { double radius; };
struct Rectangle { double w, h; };

using Shape = std::variant<Circle, Rectangle>;

struct AreaVisitor {
    double operator()(const Circle& c) { return 3.14159 * c.radius * c.radius; }
    double operator()(const Rectangle& r) { return r.w * r.h; }
};

int main() {
    std::vector<Shape> shapes;
    shapes.push_back(Circle{5.0});
    shapes.push_back(Rectangle{4.0, 6.0});

    for (const auto& s : shapes) {
        // Using std::visit
        double area = std::visit(AreaVisitor{}, s);
        std::cout << "Area: " << area << "\n";
        
        // Using holds_alternative / get_if
        if (auto c = std::get_if<Circle>(&s)) {
            std::cout << "  is Circle with r=" << c->radius << "\n";
        }
    }

    return 0;
}
