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

    const std::vector<float> lshEmbeddings{
         2.0F,  1.0F,
        -2.0F,  1.0F,
        -2.0F, -1.0F,
         1.0F,  1.0F,
         1.0F, -1.0F
    };

    const auto table = lsh_builder(lshEmbeddings, 2, hyperplanes);
    assert(table.size() == 4);
    assert((table.at(7) == std::vector<std::size_t>{0, 3}));
    assert((table.at(10) == std::vector<std::size_t>{1}));
    assert((table.at(8) == std::vector<std::size_t>{2}));
    assert((table.at(13) == std::vector<std::size_t>{4}));

    bool invalidEmbeddingsThrown = false;
    try {
        lsh_builder({1.0F, 2.0F, 3.0F}, 2, hyperplanes);
    } catch (const std::invalid_argument&) {
        invalidEmbeddingsThrown = true;
    }
    assert(invalidEmbeddingsThrown);

    const auto generatedA = hyperPlaneGen(4, 3, 12345);
    const auto generatedB = hyperPlaneGen(4, 3, 12345);

    assert(generatedA.size() == 4);
    for (const auto& plane : generatedA) {
        assert(plane.size() == 3);
    }
    assert(generatedA == generatedB);

    const auto rejectsHyperplaneParameters = [](int b, int d) {
        try {
            hyperPlaneGen(b, d, 12345);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };

    assert(rejectsHyperplaneParameters(0, 3));
    assert(rejectsHyperplaneParameters(65, 3));
    assert(rejectsHyperplaneParameters(4, 0));

    const auto indexA = index_builder(lshEmbeddings, 2, 3, 4, 12345);
    const auto indexB = index_builder(lshEmbeddings, 2, 3, 4, 12345);

    assert(indexA.d == 2);
    assert(indexA.collection.size() == 3);

    for (std::size_t i = 0; i < indexA.collection.size(); ++i) {
        const auto& tableA = indexA.collection[i];
        const auto& tableB = indexB.collection[i];

        assert(tableA.hyperplanes.size() == 4);
        assert(tableA.hyperplanes == tableB.hyperplanes);
        assert(tableA.lsh_map == tableB.lsh_map);

        std::size_t indexedDocuments = 0;
        for (const auto& bucket : tableA.lsh_map) {
            indexedDocuments += bucket.second.size();
        }
        assert(indexedDocuments == 5);
    }

    assert(indexA.collection[0].hyperplanes !=
           indexA.collection[1].hyperplanes);

    bool invalidTableCountThrown = false;
    try {
        index_builder(lshEmbeddings, 2, 0, 4, 12345);
    } catch (const std::invalid_argument&) {
        invalidTableCountThrown = true;
    }
    assert(invalidTableCountThrown);

    const lsh_index_t candidateIndex{
        2,
        4,
        {
            {hyperplanes, {{7, {0, 1}}}},
            {hyperplanes, {{7, {1, 2}}}},
            {hyperplanes, {{8, {3}}}}
        }
    };

    const auto candidates = candidateCollect(candidateIndex, {1.0F, 0.0F});
    assert((candidates == std::vector<std::size_t>{0, 1, 2}));

    assert(candidateCollect(candidateIndex, {-1.0F, 0.0F}).empty());

    bool invalidCandidateDimensionThrown = false;
    try {
        candidateCollect(candidateIndex, {1.0F});
    } catch (const std::invalid_argument&) {
        invalidCandidateDimensionThrown = true;
    }
    assert(invalidCandidateDimensionThrown);

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

    const auto rankedCandidates =
        CandidateRank(query, embeddings, 2, {2, 1, 3}, 2);

    assert(rankedCandidates.size() == 2);
    assert(rankedCandidates[0].id == 3 && rankedCandidates[0].score == 1.0F);
    assert(rankedCandidates[1].id == 1 && rankedCandidates[1].score == 0.0F);

    const auto candidateTie =
        CandidateRank(query, embeddings, 2, {3, 0}, 1);

    assert(candidateTie.size() == 1);
    assert(candidateTie[0].id == 0);

    const auto moreKThanCandidates =
        CandidateRank(query, embeddings, 2, {1, 2}, 5);

    assert(moreKThanCandidates.size() == 2);
    assert(moreKThanCandidates[0].id == 1);
    assert(moreKThanCandidates[1].id == 2);

    assert(CandidateRank(query, embeddings, 2, {}, 2).empty());

    bool invalidCandidateIdThrown = false;
    try {
        CandidateRank(query, embeddings, 2, {4}, 2);
    } catch (const std::invalid_argument&) {
        invalidCandidateIdThrown = true;
    }
    assert(invalidCandidateIdThrown);

    const auto lshResults =
    lshSearch(query, embeddings, candidateIndex, 2);

    assert(lshResults.size() == 2);
    assert(lshResults[0].id == 0 && lshResults[0].score == 1.0F);
    assert(lshResults[1].id == 1 && lshResults[1].score == 0.0F);

    assert(lshSearch({-1.0F, 0.0F}, embeddings, candidateIndex, 2).empty());
    

}
