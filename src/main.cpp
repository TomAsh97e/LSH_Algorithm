#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "lsh_index.hpp"
#include "search.hpp"

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

    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> embeddings{
         1.0F, 0.0F,  
         0.0F, 1.0F,  
        -1.0F, 0.0F,  
         1.0F, 0.0F   
    };

    const auto top2 = exactSearch(query, embeddings, 2, 2);
    assert(top2.size() == 2);
    assert(top2[0].id == 0 && top2[0].score == 1.0F);
    assert(top2[1].id == 3 && top2[1].score == 1.0F);

    const auto all = exactSearch(query, embeddings, 2, 10);
    assert(all.size() == 4);
    assert(all[0].id == 0);
    assert(all[1].id == 3);
    assert(all[2].id == 1);
    assert(all[3].id == 2);

    assert(exactSearch(query, {}, 2, 2).empty());

    bool invalidDimensionThrown = false;
    try {
        exactSearch({1.0F}, embeddings, 2, 2);
    } catch (const std::invalid_argument&) {
        invalidDimensionThrown = true;
    }
    assert(invalidDimensionThrown);



}
