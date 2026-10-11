#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "data_loader.hpp"
#include "lsh_index.hpp"
#include "search.hpp"

void printResults(
    const char* name,
    const std::vector<SearchResult>& results
) {
    std::cout << name << '\n';

    for (const auto& result : results) {
        std::cout
            << "id: " << result.id
            << ", score: " << result.score
            << '\n';
    }
}

int main() {
    try {
        const auto documents = load("data/documents.bin");
        const auto query = load("data/query.bin");

        if (query.count != 1) {
            throw std::runtime_error(
                "query.bin must contain exactly one embedding"
            );
        }

        if (documents.d != query.d) {
            throw std::runtime_error(
                "document and query dimensions are different"
            );
        }

        const int d = static_cast<int>(documents.d);
        constexpr int k = 3;

        const auto index = index_builder(
            documents.embeddings,
            d,
            8,
            4,
            12345
        );

        const auto exactResults = exactSearch(
            query.embeddings,
            documents.embeddings,
            d,
            k
        );

        const auto lshResults = lshSearch(
            query.embeddings,
            documents.embeddings,
            index,
            k
        );

        printResults("Exact search:", exactResults);
        printResults("LSH search:", lshResults);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
