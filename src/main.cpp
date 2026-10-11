#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_loader.hpp"
#include "lsh_index.hpp"
#include "search.hpp"

std::vector<std::string> loadLines(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    if (file.bad()) {
        throw std::runtime_error("cannot read file: " + path);
    }

    return lines;
}

void printResults(const char* name,const std::vector<SearchResult>& results,
    const std::vector<std::string>& documentTexts, const std::vector<std::string>& documentIds) {
    
    std::cout << '\n' << name << '\n';

    for (std::size_t rank = 0; rank < results.size(); ++rank) {
        const auto& result = results[rank];

        std::cout
            << rank + 1 << ". "
            << "dataset id: " << documentIds[result.id]
            << ", score: " << result.score << '\n'
            << documentTexts[result.id] << "\n\n";
    }
}

int main() {
    try {
        const auto documents = load("data/documents.bin");
        const auto query = load("data/query.bin");

        const auto documentTexts =
            loadLines("data/documents.txt");

        const auto documentIds =
            loadLines("data/document_ids.txt");

        const auto queryTexts =
            loadLines("data/query.txt");

        if (query.count != 1 || queryTexts.size() != 1) {
            throw std::runtime_error(
                "query files must contain exactly one query"
            );
        }

        if (documents.d != query.d) {
            throw std::runtime_error(
                "document and query dimensions are different"
            );
        }

        if (documentTexts.size() != documents.count ||
            documentIds.size() != documents.count) {
            throw std::runtime_error(
                "document files have different sizes"
            );
        }

        constexpr int k = 3;
        constexpr int tables = 8;
        constexpr int bits = 4;
        constexpr std::uint32_t seed = 12345;

        const int d = static_cast<int>(documents.d);

        std::cout
            << "Documents: " << documents.count << '\n'
            << "Query: " << queryTexts[0] << '\n';

        const auto index = index_builder(
            documents.embeddings,
            d,
            tables,
            bits,
            seed
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

        printResults(
            "Exact search:",
            exactResults,
            documentTexts,
            documentIds
        );

        printResults(
            "LSH search:",
            lshResults,
            documentTexts,
            documentIds
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}