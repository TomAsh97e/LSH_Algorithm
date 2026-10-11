#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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
        struct Config {
            int tables;
            int bits;
        };

        constexpr std::size_t documentCounts[] = {1000, 10000, 25657};
        constexpr Config configs[] = {{8, 8}, {16, 8}, {16, 6}};
        constexpr std::uint32_t seeds[] = {12345, 23456, 34567};
        constexpr int k = 10;
        constexpr std::size_t warmupCount = 5;

        const auto documents = load("data/documents.bin");
        const auto queries = load("data/queries_test.bin");
        const std::size_t queryCount = queries.count;

        if (documents.d != queries.d) throw std::runtime_error("dimension mismatch");
        if (queries.count < warmupCount) throw std::runtime_error("not enough queries");

        const int d = static_cast<int>(documents.d);
        using Clock = std::chrono::steady_clock;

        std::vector<std::vector<float>> queryVectors;
        queryVectors.reserve(queryCount);

        for (std::size_t queryId = 0; queryId < queryCount; ++queryId) {
            const std::size_t start = queryId * queries.d;
            queryVectors.emplace_back(queries.embeddings.begin() + start, queries.embeddings.begin() + start + queries.d);
        }

        std::ofstream csv("benchmark/test_results.csv");
        if (!csv) throw std::runtime_error("cannot open CSV");
        csv << "N,L,b,seed,query_id,exact_us,lsh_us,candidates,recall,build_ms\n";

        for (const std::size_t documentCount : documentCounts) {
            if (documents.count < documentCount) throw std::runtime_error("not enough documents");

            const std::vector<float> sampledDocuments(
                documents.embeddings.begin(), documents.embeddings.begin() + documentCount * documents.d);

            for (std::size_t i = 0; i < warmupCount; ++i) exactSearch(queryVectors[i], sampledDocuments, d, k);

            std::vector<std::vector<SearchResult>> exactResults;
            std::vector<double> exactTimesUs;
            exactResults.reserve(queryCount);
            exactTimesUs.reserve(queryCount);
            double exactTotalUs = 0.0;

            for (const auto& query : queryVectors) {
                const auto start = Clock::now();
                const auto result = exactSearch(query, sampledDocuments, d, k);
                const auto stop = Clock::now();
                if (result.size() != static_cast<std::size_t>(k)) throw std::runtime_error("wrong exact result count");

                const double exactUs = std::chrono::duration<double, std::micro>(stop - start).count();
                exactTotalUs += exactUs;
                exactTimesUs.push_back(exactUs);
                exactResults.push_back(result);
            }

            const double exactAverageUs = exactTotalUs / queryCount;

            for (const std::uint32_t seed : seeds) {
                for (const Config config : configs) {
                    const auto buildStart = Clock::now();
                    const auto index = index_builder(sampledDocuments, d, config.tables, config.bits, seed);
                    const auto buildStop = Clock::now();
                    const double buildMs = std::chrono::duration<double, std::milli>(buildStop - buildStart).count();

                    for (std::size_t i = 0; i < warmupCount; ++i) {
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

                        const double lshUs = std::chrono::duration<double, std::micro>(stop - start).count();
                        const double recall = recallAtK(exactResults[queryId], lshResults, k);
                        lshTotalUs += lshUs;
                        recallSum += recall;
                        candidateSum += candidates.size();

                        csv << documentCount << ',' << config.tables << ',' << config.bits << ',' << seed << ','
                            << queryId << ',' << exactTimesUs[queryId] << ',' << lshUs << ',' << candidates.size()
                            << ',' << recall << ',' << buildMs << '\n';
                    }

                    const double averageCandidates = static_cast<double>(candidateSum) / queryCount;
                    const double averageRecall = recallSum / queryCount;
                    const double lshAverageUs = lshTotalUs / queryCount;

                    std::cout << "N=" << documentCount << ", L=" << config.tables << ", b=" << config.bits
                              << ", seed=" << seed << ", candidates=" << averageCandidates
                              << ", recall=" << averageRecall << ", exact_us=" << exactAverageUs
                              << ", lsh_us=" << lshAverageUs << ", build_ms=" << buildMs << '\n';
                }
            }
        }

        if (!csv) throw std::runtime_error("CSV write failed");
        std::cout << "Saved benchmark/test_results.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
