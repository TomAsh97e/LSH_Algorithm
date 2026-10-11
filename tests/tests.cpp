#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "data_loader.hpp"
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

    std::vector<std::vector<float>> sixtyFourHyperplanes(
        64,
        std::vector<float>{-1.0F}
    );
    sixtyFourHyperplanes[0][0] = 1.0F;
    sixtyFourHyperplanes[63][0] = 1.0F;

    const auto firstAndLastBit =
        (std::uint64_t{1} << 0) | (std::uint64_t{1} << 63);
    assert(signatureFromVector({1.0F}, sixtyFourHyperplanes) ==
           firstAndLastBit);

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

    const auto top1 = exactSearch(query, embeddings, 2, 1);
    assert(top1.size() == 1);
    assert(top1[0].id == 0 && top1[0].score == 1.0F);

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

    const auto loaderTestFile =
        std::filesystem::temp_directory_path() / "lsh_embeddings_test.bin";

    const std::uint64_t fileN = 2;
    const std::uint64_t fileD = 3;
    const std::vector<float> fileValues{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F
    };

    {
        std::ofstream output(loaderTestFile, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&fileN), sizeof(fileN));
        output.write(reinterpret_cast<const char*>(&fileD), sizeof(fileD));
        output.write(
            reinterpret_cast<const char*>(fileValues.data()),
            fileValues.size() * sizeof(float)
        );
        assert(output);
    }

    const auto loaded = load(loaderTestFile);

    assert(loaded.count == fileN);
    assert(loaded.d == fileD);
    assert(loaded.embeddings == fileValues);

    {
        std::ofstream output(loaderTestFile, std::ios::binary | std::ios::app);
        const char extraByte = 1;
        output.write(&extraByte, sizeof(extraByte));
    }

    bool extraDataThrown = false;
    try {
        load(loaderTestFile);
    } catch (const std::runtime_error&) {
        extraDataThrown = true;
    }
    assert(extraDataThrown);

    const auto writeTestEmbedding = [&loaderTestFile](
        std::uint64_t n,
        std::uint64_t d,
        const std::vector<float>& values
    ) {
        std::ofstream output(
            loaderTestFile,
            std::ios::binary | std::ios::trunc
        );
        output.write(reinterpret_cast<const char*>(&n), sizeof(n));
        output.write(reinterpret_cast<const char*>(&d), sizeof(d));
        output.write(
            reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(float)
        );
        assert(output);
    };

    const auto loadFails = [&loaderTestFile]() {
        try {
            load(loaderTestFile);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };

    writeTestEmbedding(2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
    const bool truncatedFileRejected = loadFails();

    writeTestEmbedding(
        1,
        2,
        {std::numeric_limits<float>::quiet_NaN(), 1.0F}
    );
    const bool nanRejected = loadFails();

    writeTestEmbedding(
        1,
        2,
        {std::numeric_limits<float>::infinity(), 1.0F}
    );
    const bool infinityRejected = loadFails();

    writeTestEmbedding(1, 2, {0.0F, 0.0F});
    const bool zeroVectorRejected = loadFails();

    std::filesystem::remove(loaderTestFile);

    assert(truncatedFileRejected);
    assert(nanRejected);
    assert(infinityRejected);
    assert(zeroVectorRejected);

}
