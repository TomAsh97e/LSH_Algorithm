#include "data_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <limits>
#include <cmath>

embeddingData load (const std::filesystem::path& source){

    std::ifstream file(source, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("file problems");
    }
    std::uint64_t n{};
    std::uint64_t d{};

    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    file.read(reinterpret_cast<char*>(&d), sizeof(d));

    if(!file){
        throw std::runtime_error("can't read n or d");
    }
    if(n == 0 || d == 0 || n > std::numeric_limits<std::size_t>::max() / d) throw std::invalid_argument("n or d == 0 or n too big");
    std::vector<float> values(n*d);
    file.read(reinterpret_cast<char*>(values.data()),values.size() * sizeof(float));

    if(!file){
        throw std::runtime_error("can't read floats");
    }

    if(file.peek() != std::ifstream::traits_type::eof()){
        throw std::runtime_error("more data after floats");
    }
    const std::size_t dimension = static_cast<std::size_t>(d);

    for (std::size_t start = 0; start < values.size(); start += dimension) {
        bool nonZero = false;

        for (std::size_t i = start; i < start + dimension; ++i) {
            if (!std::isfinite(values[i])) {
                throw std::invalid_argument("embedding contains NaN or infinity");
            }

            if (values[i] != 0.0F) {
            nonZero = true;
            }
        }

        if (!nonZero) {
            throw std::invalid_argument("embedding is a zero vector");
        }
    }  
    embeddingData res{n, d, std::move(values)};
    return res;
}