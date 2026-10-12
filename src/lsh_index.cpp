#include "lsh_index.hpp"
#include <stdexcept>
#include <bitset>
#include <random>

uint64_t signatureFromVector(const std::vector<float>& documentVector, const std::vector<std::vector<float>>& hyperplanes){
    uint64_t res = 0;
    if(hyperplanes.size() > 64 || hyperplanes.size() < 1) throw std::invalid_argument("More than 64 hyperplanes or less than 1");
    std::bitset<64> bits(res);
    for(size_t i =0; i < hyperplanes.size(); i++){
        if(dotToBit(dot(documentVector, hyperplanes[i])) == true){
            bits.set(i);
        }
    }
    res = bits.to_ullong();
    return res;
}

uint64_t signatureFromVector_perf(const std::vector<float>& embeddings, const std::vector<std::vector<float>>& hyperplanes, std::size_t n, int d){
    uint64_t res = 0;
    if(hyperplanes.size() > 64 || hyperplanes.size() < 1) throw std::invalid_argument("More than 64 hyperplanes or less than 1");
    std::bitset<64> bits(res);
    for(size_t i =0; i < hyperplanes.size(); i++){
        if(dotToBit(dot_perf(embeddings, hyperplanes[i], n ,d))){
            bits.set(i);
        }
    }
    res = bits.to_ullong();
    return res;
}

float dot(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size() != b.size()) throw std::invalid_argument("Vectors have differnt siezes");
    if(a.size() == 0) throw std::invalid_argument("Vectors have size 0"); 
    float result = 0;
    for(size_t i = 0; i < a.size(); i++){
        result += a[i]*b[i];
    }
    return result;
}
float dot_perf(const std::vector<float>& embeddings, const std::vector<float>& asked, std::size_t i, int d){
    float res{};
    const std::size_t start = i* static_cast<std::size_t>(d);
    const float* document = embeddings.data() + start;
    const float* query = asked.data();
    for(int k = 0; k < d; k++){
        res += document[k] * query[k];
    }
    return res;
}


bool dotToBit(float dotProduct){
    return dotProduct >= 0 ? true : false;
}

std::unordered_map<uint64_t, std::vector<std::size_t>> lsh_builder (const std::vector<float>& embeddings, int d,
    const std::vector<std::vector<float>>& hyperplanes){
        if(d <= 0  || embeddings.size() % d != 0) {
            throw std::invalid_argument("d not > 0 or d or embeddings values are worng");}
        int docsCount = embeddings.size()/d;
        std::unordered_map<uint64_t, std::vector<std::size_t>> res;
        for (const auto& hyperplane : hyperplanes) {
            if (hyperplane.size() != static_cast<std::size_t>(d)) {
                throw std::invalid_argument("wrong hyperplane dimension");
            }
        }
        for(int i = 0; i < docsCount; i++){
            const auto sig = signatureFromVector_perf(embeddings, hyperplanes, i, d);
            res[sig].push_back(i);
        }
        return res;
    }

std::vector<std::vector<float>> hyperPlaneGen(int b, int d, std::uint32_t seed){
    if(d <= 0 || b <= 0 || b > 64) throw std::invalid_argument("d or b worng values");
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0,1);
    std::vector<std::vector<float>> res(b, std::vector<float> (d));
    for(size_t i =0; i < b; i++ ){
        for(size_t k =0; k < d; k++){
            res[i][k] = dist(gen);
        }
    }
    return res;
}


lsh_index_t index_builder(const std::vector<float>& embeddings, int d, int l, int b, 
    std::uint32_t seed){
        if(l <= 0) throw std::invalid_argument("wrong l value");
        lsh_index_t res{};
        res.collection.reserve(l);
        res.d = d;
        if(d <= 0 || embeddings.size() % d != 0) throw std::invalid_argument("wrong d value");
        res.documentCount = embeddings.size()/d;
        for(int i = 0; i < l; i++){
            lsh_t tmp{};
            tmp.hyperplanes = hyperPlaneGen(b, d, seed + i);
            tmp.lsh_map = lsh_builder(embeddings, d, tmp.hyperplanes);
            res.collection.push_back(std::move(tmp));
        }
        return res;

}


std::vector<std::size_t> candidateCollect(const lsh_index_t& index, 
    const std::vector<float>& asked){
        if(asked.size() != index.d) throw std::invalid_argument("wrog d");
        std::vector<std::size_t> res{};
        std::vector<std::uint8_t>seen(index.documentCount, 0);
        for(const auto& a : index.collection){
            const auto sig = signatureFromVector(asked, a.hyperplanes);
            const auto it  = a.lsh_map.find(sig);
            if(it == a.lsh_map.end()){continue;} 

            const auto& ids = it -> second;

            for(const auto& id : ids){
                if(seen[id] != 0){continue;}

                seen[id] = 1;
                res.push_back(id);
                }
            }
    return res;
}