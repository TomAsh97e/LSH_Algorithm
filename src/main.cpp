#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include "lsh_index.hpp"

int main() {
    const std::vector<std::vector<float>> hyperplanes{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, -1.0F},
        {-1.0F, -1.0F}
    };

    const std::vector<std::vector<float>> documents{
        {2.0F, 1.0F},
        {-2.0F, 1.0F},
        {-2.0F, -1.0F},
        {1.0F, 1.0F},
        {1.0F, -1.0F}
    };
    const std::vector<std::uint64_t> expected{7, 10, 8, 7, 13};

    for (std::size_t i = 0; i < documents.size(); ++i) {
        assert(signatureFromVector(documents[i], hyperplanes) == expected[i]);
        std::cout << signatureFromVector(documents[i], hyperplanes) << std::endl;
    }


}