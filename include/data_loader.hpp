#pragma once
#include <iostream>
#include <vector>
#include <filesystem>

struct embeddingData{
    size_t count;
    size_t d;
    std::vector<float> embeddings;
};

embeddingData load (const std::filesystem::path& source);