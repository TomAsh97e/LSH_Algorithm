#pragma once
#include <cstdint>
#include <vector>

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