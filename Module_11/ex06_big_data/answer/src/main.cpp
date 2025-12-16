#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>

// Simulated MapReduce

// Mapper: Sentence -> List of (Word, 1)
std::vector<std::pair<std::string, int>> mapper(const std::string& document) {
    std::vector<std::pair<std::string, int>> results;
    std::stringstream ss(document);
    std::string word;
    while (ss >> word) {
        // Simple normalization
        results.push_back({word, 1});
    }
    return results;
}

// Reducer: Word, List of Counts -> Total Count
int reducer(const std::string& key, const std::vector<int>& values) {
    int sum = 0;
    for (int v : values) sum += v;
    return sum;
}

int main() {
    std::vector<std::string> dataset = {
        "hello world",
        "hello C++",
        "C++ is fast",
        "distributed computing with map reduce"
    };

    std::cout << "--- Map Phase ---" << std::endl;
    std::vector<std::pair<std::string, int>> intermediate;
    for (const auto& doc : dataset) {
        auto pairs = mapper(doc);
        intermediate.insert(intermediate.end(), pairs.begin(), pairs.end());
    }

    std::cout << "Generated " << intermediate.size() << " pairs." << std::endl;

    std::cout << "--- Shuffle/Sort Phase ---" << std::endl;
    std::map<std::string, std::vector<int>> grouped;
    for (const auto& p : intermediate) {
        grouped[p.first].push_back(p.second);
    }

    std::cout << "--- Reduce Phase ---" << std::endl;
    for (const auto& entry : grouped) {
        int count = reducer(entry.first, entry.second);
        std::cout << entry.first << ": " << count << std::endl;
    }

    return 0;
}
