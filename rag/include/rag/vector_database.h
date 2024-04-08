#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ds
{
using Embedding = std::vector<float>;

struct RetrievedIndex
{
    size_t index;
    float cosine_similarity;
};

class IVectorStore
{
  public:
    virtual ~IVectorStore() = default;

    virtual void add(const std::vector<Embedding>& chunks) = 0;
    virtual std::vector<RetrievedIndex> retrieve(const std::vector<float>& values, uint32_t top_k) = 0;
};

std::unique_ptr<IVectorStore> vector_store_factory(size_t embedding_rank);
} // namespace ds