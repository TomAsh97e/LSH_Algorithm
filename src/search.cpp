#include "search.hpp"
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

std::vector<SearchResult> CandidateRank(const std::vector<float>& asked, const std::vector<float>& embeddings,
    int d, std::vector<size_t> candidates, int k){
        if(d <= 0 || k <= 0 || asked.size() != d || embeddings.size() % d != 0) {
            throw std::invalid_argument("d or k not > 0 or d or embeddings values are worng");}
        std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult>> minHeap;
        for(const auto & candidate : candidates){
            if(candidate >= embeddings.size()/d) throw std::invalid_argument("candidate id too big");
            SearchResult tmp{};
            tmp.id = candidate;
            std::vector<float> sub (embeddings.begin()+candidate*d, embeddings.begin() + candidate*d + d);
            tmp.score = dot(sub, asked);
            if(minHeap.size() < k){minHeap.push(tmp);}
            else if(tmp > minHeap.top()){
                minHeap.pop();
                minHeap.push(tmp);
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

std::vector<SearchResult> lshSearch(const std::vector<float>& asked, const std::vector<float>& embeddings,
    const lsh_index_t& index , int k){
        return CandidateRank(asked, embeddings,
            index.d ,candidateCollect(index, asked), k);
    }