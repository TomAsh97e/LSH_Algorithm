#include "data_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <limits>

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
    embeddingData res{n, d, std::move(values)};
    return res;
}