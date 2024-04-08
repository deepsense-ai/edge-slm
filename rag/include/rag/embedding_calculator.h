#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

namespace ds
{
struct EmbeddingCalculatorParams
{
    std::filesystem::path model_path;
    int32_t n_threads;
    int32_t batch_size;
};

using Embedding = std::vector<float>;

struct EmbeddingCalculationResult {
  Embedding embedding;
  size_t n_tokens;
};

class IEmbeddingCalculator
{
  public:
    virtual ~IEmbeddingCalculator() = default;

    virtual EmbeddingCalculationResult calc(const std::string& chunk) const;
    virtual std::vector<EmbeddingCalculationResult> calc_batch(const std::vector<std::string>& chunks) const = 0;
    virtual size_t get_embedding_rank() const = 0;
};

std::unique_ptr<IEmbeddingCalculator> embedding_calculator_factory(const EmbeddingCalculatorParams& params);
} // namespace ds