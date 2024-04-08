#include "rag/vector_database.h"

#include <algorithm>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace ds
{
class FaissIndexDatabase : public IVectorStore
{
  public:
    explicit FaissIndexDatabase(size_t embedding_rank) : embedding_rank_(embedding_rank), index_(embedding_rank) {}

    void add(const std::vector<Embedding>& embeddings) override
    {
        if(!check_embeddings_size_(embeddings))
        {
            throw std::logic_error(fmt::format("Invalid values size, expected rank: {}", embedding_rank_));
        }

        std::vector<float> embeddings_flattened;
        embeddings_flattened.reserve(embeddings.size() * embedding_rank_);
        std::for_each(embeddings.begin(), embeddings.end(),
                      [&embeddings_flattened](const auto& embedding)
                      { embeddings_flattened.insert(embeddings_flattened.end(), embedding.begin(), embedding.end()); });

        float* data_to_add = embeddings_flattened.data();
        size_t num_vectors_to_add = embeddings.size();

        // L2 norm all vector embeddings in batch.
        faiss::fvec_renorm_L2(embedding_rank_, num_vectors_to_add, data_to_add);
        index_.add(num_vectors_to_add, data_to_add);
    }

    std::vector<RetrievedIndex> retrieve(const std::vector<float>& values, uint32_t top_k) override
    {
        if(values.size() != embedding_rank_)
        {
            throw std::logic_error(fmt::format("Invalid values size, expected rank: {}", embedding_rank_));
        }

        std::vector<RetrievedIndex> retrieved_indices;
        retrieved_indices.reserve(top_k);
        std::vector<float> distances(top_k);
        std::vector<faiss::idx_t> labels(top_k);

        // Copy the vector, we must L2 normalize the vector before performing search.
        std::vector<float> embeddings_cpy(values);

        // The magic "1" is just to indicate the API that we're launching lookup for only one values vector
        // The faiss API can also lookup for multiple embeddings at the same call (in batch basically).
        // But it's not our case yet.
        faiss::fvec_renorm_L2(embedding_rank_, 1, embeddings_cpy.data());
        index_.search(1, embeddings_cpy.data(), static_cast<faiss::idx_t>(top_k), distances.data(), labels.data());

        // last step - retrieval of the actual chunks.
        std::transform(labels.begin(), labels.end(), distances.begin(), std::back_inserter(retrieved_indices),
                       [this](const faiss::idx_t& label, const float& distance) -> RetrievedIndex
                       {
                           const auto idx = static_cast<size_t>(label);
                           return RetrievedIndex{idx, distance};
                       });

        return retrieved_indices;
    }

  private:
    size_t embedding_rank_;
    faiss::IndexFlatIP index_;

    bool check_embeddings_size_(const std::vector<Embedding>& embeddings) const
    {
        return std::all_of(embeddings.begin(), embeddings.end(),
                           [this](const auto& embedding) -> bool { return embedding.size() == embedding_rank_; });
    }
};

std::unique_ptr<IVectorStore> vector_store_factory(size_t embedding_rank)
{
    return std::make_unique<FaissIndexDatabase>(embedding_rank);
}

} // namespace ds
