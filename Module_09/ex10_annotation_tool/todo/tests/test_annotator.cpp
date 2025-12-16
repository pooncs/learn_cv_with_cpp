#include <gtest/gtest.h>
#include <QApplication>
#include <QVector>
#include <QRect>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

TEST(Annotation, JsonSerialization) {
    QVector<QRect> boxes;
    boxes.append(QRect(10, 10, 100, 100));
    boxes.append(QRect(50, 50, 20, 20));

    json j_boxes = json::array();
    for (const auto &box : boxes) {
        j_boxes.push_back({
            {"x", box.x()},
            {"y", box.y()},
            {"w", box.width()},
            {"h", box.height()}
        });
    }

    std::string dump = j_boxes.dump();
    EXPECT_FALSE(dump.empty());
    
    auto j_parsed = json::parse(dump);
    EXPECT_EQ(j_parsed.size(), 2);
    EXPECT_EQ(j_parsed[0]["x"], 10);
    EXPECT_EQ(j_parsed[1]["w"], 20);
}
