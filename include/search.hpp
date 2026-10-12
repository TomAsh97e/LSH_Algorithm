#pragma once
#include <cstdint>
#include <vector>
#include "lsh_index.hpp"

struct SearchResult {
    std::size_t id;
    float score;

    bool operator>(const SearchResult& other) const{
        if(score != other.score){
            return score > other.score;
        }
        return id < other.id;
    }
};

std::vector<SearchResult> exactSearch(const std::vector<float>& asked, const std::vector<float>& embeddings, int d, int k);


std::vector<SearchResult> CandidateRank(const std::vector<float>& asked, const std::vector<float>& embeddings,
    int d, const std::vector<size_t>& candidates, int k);

std::vector<SearchResult> lshSearch(const std::vector<float>& asked, const std::vector<float>& embeddings,
    const lsh_index_t& index , int k);