#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>

#include "data_loader.hpp"
#include "lsh_index.hpp"
#include "search.hpp"

double recallAtK(const std::vector<SearchResult>& exactResults, const std::vector<SearchResult>& lshResults, int k) {
    std::size_t matches = 0;

    for (const auto& exact : exactResults) {
        for (const auto& approximate : lshResults) {
            if (exact.id == approximate.id) {
                ++matches;
                break;
            }
        }
    }

    return static_cast<double>(matches) / k;
}

int main() {
    try {
        constexpr std::size_t documentCount = 10000;
        constexpr int k = 10;
        constexpr int tableCounts[] = {4, 8, 16};
        constexpr int bitCounts[] = {6, 8, 10};
        constexpr std::uint32_t seed = 12345;
        constexpr std::size_t queryCount = 100;

        const auto documents = load("data/documents.bin");
        const auto queries = load("data/queries_tuning.bin");

        if (documents.d != queries.d) throw std::runtime_error("dimension mismatch");
        if (documents.count < documentCount) throw std::runtime_error("not enough documents");
        if (queries.count < queryCount) throw std::runtime_error("not enough queries");

        const int d = static_cast<int>(documents.d);
        const std::vector<float> sampledDocuments(documents.embeddings.begin(),
                documents.embeddings.begin() + documentCount * documents.d);

        using Clock = std::chrono::steady_clock;

        std::vector<std::vector<float>> queryVectors;
        queryVectors.reserve(queryCount);

        for (std::size_t queryId = 0; queryId < queryCount; ++queryId) {
            const std::size_t start = queryId * queries.d;
            queryVectors.emplace_back(queries.embeddings.begin() + start,
                        queries.embeddings.begin() + start + queries.d);
        }

        constexpr std::size_t warmupCount = 5;

        for (std::size_t i = 0; i < warmupCount; ++i) {
            exactSearch(queryVectors[i], sampledDocuments, d, k);
        }
        std::vector<std::vector<SearchResult>> exactResults;
        exactResults.reserve(queryCount);

        double exactTotalUs = 0.0;

        for (const auto& query : queryVectors) {
            const auto start = Clock::now();
            const auto result = exactSearch(query, sampledDocuments, d, k);
            const auto stop = Clock::now();
            if (result.size() != static_cast<std::size_t>(k)){
                throw std::runtime_error("wrong exact result count");}
            exactTotalUs += std::chrono::duration<double, std::micro>(stop - start).count();
            exactResults.push_back(result);
        }

        const double exactAverageUs = exactTotalUs / queryCount;

        for (const int tables : tableCounts) {
            for (const int bits : bitCounts) {
                const auto buildStart = Clock::now();
                const auto index = index_builder(sampledDocuments, d, tables, bits, seed);
                const auto buildStop = Clock::now();

                const double buildMs = std::chrono::duration<double, std::milli>(buildStop - buildStart).count();
                for(std::size_t i = 0; i < warmupCount; i++){
                    const auto candidates = candidateCollect(index, queryVectors[i]);
                    CandidateRank(queryVectors[i], sampledDocuments, d, candidates, k);
                }

                double lshTotalUs = 0.0;
                double recallSum = 0.0;
                std::size_t candidateSum = 0;

                for (std::size_t queryId = 0; queryId < queryCount; ++queryId) {

                    const auto start = Clock::now();

                    const auto candidates = candidateCollect(index, queryVectors[queryId]);
                    const auto lshResults = CandidateRank(queryVectors[queryId], sampledDocuments, d, candidates, k);
                    
                    const auto stop = Clock::now();
                    if (lshResults.size() > static_cast<std::size_t>(k)) throw std::runtime_error("too many LSH results");
                    if (candidates.size() > documentCount) throw std::runtime_error("too many candidates");
                    
                    lshTotalUs += std::chrono::duration<double, std::micro>(stop - start).count();
                    recallSum += recallAtK(exactResults[queryId], lshResults, k);
                    candidateSum += candidates.size();
                }
                    const double averageCandidates = static_cast<double>(candidateSum)/ queryCount;
                    const double averageRecall = recallSum / queryCount;
                    const double lshAverageUs = lshTotalUs / queryCount;
                    std::cout << "L=" << tables << ", b=" << bits << ", candidates=" << averageCandidates
                    << ", recall=" << averageRecall << ", exact_us=" << exactAverageUs
                    << ", lsh_us=" << lshAverageUs << ", build_ms=" << buildMs << '\n'; 
                
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
