#include "lsh_index.hpp"
#include <stdexcept>
#include <bitset>

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

float dot(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size() != b.size()) throw std::invalid_argument("Vectors have differnt siezes");
    if(a.size() == 0) throw std::invalid_argument("Vectors have size 0"); 
    float result = 0;
    for(size_t i = 0; i < a.size(); i++){
        result += a[i]*b[i];
    }
    return result;
}

bool dotToBit(float dotProduct){
    return dotProduct >= 0 ? true : false;
}