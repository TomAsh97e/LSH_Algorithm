#include "search.hpp"
#include "lsh_index.hpp"
#include "stdexcept"
#include <queue>
#include <algorithm>

std::vector<SearchResult> exactSearch(const std::vector<float>& asked, 
    const std::vector<float>& embeddings, int d, int k){
        if(d <= 0 || k <= 0 || asked.size() != d || embeddings.size() % d != 0) {
            throw std::invalid_argument("d or k not > 0 or d or embeddings values are worng");}
        int docsCount = embeddings.size()/d;
        std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult>> minHeap;
        for(size_t i = 0; i < embeddings.size(); i += d){
            SearchResult candidate;
            candidate.id = i/d;
            std::vector<float> sub(embeddings.begin() + i, embeddings.begin() + i + d);
            candidate.score = dot(asked, sub);
            if(minHeap.size() < k){minHeap.push(candidate);}
            else if(minHeap.top().score < candidate.score){
                minHeap.pop();
                minHeap.push(candidate);
            }
        }
        std::vector<SearchResult> res {};
        while(!minHeap.empty()){
            res.push_back(minHeap.top());
            minHeap.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }