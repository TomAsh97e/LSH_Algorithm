#pragma once
#include <cstdint>
#include <vector>

uint64_t signatureFromVector(const std::vector<float>& documentVector, const std::vector<std::vector<float>> &hyperplanes);

float dot(const std::vector<float>& a, const std::vector<float>& b);

bool dotToBit(float dotProduct);
