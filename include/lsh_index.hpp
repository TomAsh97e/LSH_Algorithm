#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>

struct lsh_t{
    std::vector<std::vector<float>> hyperplanes;
    std::unordered_map<uint64_t, std::vector<std::size_t>> lsh_map;
};

struct lsh_index_t{
    int d;
    std::size_t documentCount;
    std::vector<lsh_t> collection;
};

uint64_t signatureFromVector(const std::vector<float>& documentVector, const std::vector<std::vector<float>> &hyperplanes);
uint64_t signatureFromVector_perf(const std::vector<float>& embeddings, const std::vector<std::vector<float>>& hyperplanes, std::size_t i, int d);
float dot(const std::vector<float>& a, const std::vector<float>& b);
float dot_perf(const std::vector<float>& embeddings, const std::vector<float>& asked, std::size_t i, int d);

bool dotToBit(float dotProduct);

std::unordered_map<uint64_t, std::vector<std::size_t>> lsh_builder (const std::vector<float>& embeddings, int d,
    const std::vector<std::vector<float>>& hyperplanes);

std::vector<std::vector<float>> hyperPlaneGen(int b, int d, std::uint32_t seed);


lsh_index_t index_builder(const std::vector<float>& embeddings, int d, int l, int b, 
    std::uint32_t seed);

std::vector<std::size_t> candidateCollect(const lsh_index_t& index, 
    const std::vector<float>& asked);
