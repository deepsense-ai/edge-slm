#include "rag/document_retrieval.h"
#include "llm/utils.h"

#include <algorithm>
#include <functional>
#include <nlohmann/json.hpp>
#include <ranges>
#include <spdlog/spdlog.h>

namespace ds
{
std::unique_ptr<IDocumentChunkRetriever> create_document_chunk_retriever(const DocumentChunkRetrieverParams& params)
{
    auto embedding_calculator = embedding_calculator_factory(params.embedding_calculator_params);
    auto vector_store = vector_store_factory(embedding_calculator->get_embedding_rank());

    return std::make_unique<SimpleDocumentChunkRetriever>(std::move(embedding_calculator), std::move(vector_store));
};

SimpleDocumentChunkRetriever::SimpleDocumentChunkRetriever(std::unique_ptr<IEmbeddingCalculator>&& embedding_calculator,
                                                           std::unique_ptr<IVectorStore>&& vector_store)
    : embedding_calculator_(std::move(embedding_calculator)), vector_store_(std::move(vector_store))
{
}

std::vector<RetrievedDocumentChunk> SimpleDocumentChunkRetriever::retrieve(const std::string& question,
                                                                           const size_t top_k) const
{
    const auto embedding_calculator_fn = [this, &question]() { return embedding_calculator_->calc(question); };
    const auto embedding_result = with_time_report("Query embedding", embedding_calculator_fn);

    const auto retrieve_indices_fn = [this, &embedding_result, &top_k]()
    { return vector_store_->retrieve(embedding_result.embedding, top_k); };

    const auto retrieved_indices = with_time_report("Querying vector DB", retrieve_indices_fn);

    std::vector<RetrievedDocumentChunk> output;
    output.reserve(top_k);
    std::transform(retrieved_indices.begin(), retrieved_indices.end(), std::back_inserter(output),
                   [this](const RetrievedIndex& retrieved_index) -> RetrievedDocumentChunk
                   {
                       return RetrievedDocumentChunk{document_chunks_[retrieved_index.index].content,
                                                     document_chunks_[retrieved_index.index].metadata.chunk_id,
                                                     retrieved_index.cosine_similarity};
                   });

    return output;
}

struct DocumentChunksSplit
{
    std::vector<DocumentChunk*> with_embeddings;
    std::vector<DocumentChunk*> without_embeddings;

    DocumentChunksSplit(size_t max_size)
    {
        with_embeddings.reserve(max_size);
        without_embeddings.reserve(max_size);
    }
};

DocumentChunksSplit split_document_chunks(std::vector<DocumentChunk>& chunks)
{
    DocumentChunksSplit split(chunks.size());
    std::ranges::for_each(chunks, [&split](DocumentChunk &chunk) -> void{
        if (chunk.embedding) {
            split.with_embeddings.push_back(&chunk);
        } else {
            split.without_embeddings.push_back(&chunk);
        }
    });
    return split;
}

void calculate_missing_embeddings(const IEmbeddingCalculator& embedding_calculator, DocumentChunksSplit& split) {
    auto& with_embeddings = split.with_embeddings;
    auto& without_embeddings = split.without_embeddings;

    std::vector<std::string> chunks_to_calculate_embeddings;
    std::ranges::transform(without_embeddings, std::back_inserter(chunks_to_calculate_embeddings),
                           [](const DocumentChunk* const chunk) { return chunk->content; });

    const auto calculated_embeddings = embedding_calculator.calc_batch(chunks_to_calculate_embeddings);

    for(size_t i = 0; i < calculated_embeddings.size(); i++)
    {
        auto& document_chunk = *without_embeddings[i];
        document_chunk.embedding = calculated_embeddings[i].embedding;
    }

    with_embeddings.insert(with_embeddings.end(), without_embeddings.begin(), without_embeddings.end());
    without_embeddings.clear();
}

void SimpleDocumentChunkRetriever::add_document_chunks(const std::vector<DocumentChunk>& chunks)
{
    auto chunks_cpy = chunks;
    auto splitted_chunks = split_document_chunks(chunks_cpy);

    calculate_missing_embeddings(*embedding_calculator_, splitted_chunks);

    std::vector<Embedding> embeddings_to_index;
    std::vector<DocumentChunk> chunks_to_add;

    embeddings_to_index.reserve(chunks.size());
    chunks_to_add.reserve(chunks.size());

    std::ranges::transform(splitted_chunks.with_embeddings, std::back_inserter(embeddings_to_index), [](const DocumentChunk* chunk) {
        return *chunk->embedding;
    });

    std::ranges::transform(splitted_chunks.with_embeddings, std::back_inserter(chunks_to_add), [](const DocumentChunk* chunk) {
        return *chunk;
    });

    vector_store_->add(embeddings_to_index);
    document_chunks_.insert(document_chunks_.end(), chunks_to_add.begin(), chunks_to_add.end());
}

void from_json(const nlohmann::json& j, DocumentChunkMetadata& d)
{
    j.at("source").get_to(d.source);
    j.at("chunk_id").get_to(d.chunk_id);
}

void to_json(nlohmann::json& j, const DocumentChunkMetadata& d)
{
    j = nlohmann::json{{"source", d.source}, {"chunk_id", d.chunk_id}};
}

void from_json(const nlohmann::json& j, DocumentChunk& d)
{
    j.at("content").get_to(d.content);
    j.at("metadata").get_to(d.metadata);

    if(j.count("embedding") && j.at("embedding") != nullptr)
    {
        d.embedding = std::vector<float>();
        j.at("embedding").get_to(*d.embedding);
    }
}

void to_json(nlohmann::json& j, const DocumentChunk& d)
{
    j = nlohmann::json{{"content", d.content}, {"metadata", d.metadata}};
    if(d.embedding)
        j["embedding"] = *d.embedding;
    else
        j["embedding"] = nullptr;
}

void SimpleDocumentChunkRetriever::dump(std::ostream& output) const
{
    using namespace nlohmann;
    auto object = nlohmann::json{{"chunks", document_chunks_}};

    output << object.dump();
}

void SimpleDocumentChunkRetriever::load(std::istream& input)
{
    using namespace nlohmann;
    const auto json_object = json::parse(input);
    std::vector<DocumentChunk> chunks = json_object.at("chunks");

    this->add_document_chunks(chunks);
}
} // namespace ds