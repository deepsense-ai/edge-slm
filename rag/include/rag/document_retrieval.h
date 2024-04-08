#pragma once
#include "rag/embedding_calculator.h"
#include "rag/vector_database.h"

#include <memory>
#include <optional>
#include <vector>

namespace ds
{
struct DocumentChunkRetrieverParams
{
    EmbeddingCalculatorParams embedding_calculator_params;
};

struct DocumentChunkMetadata {
    std::string source;
    uint64_t chunk_id;
};

struct DocumentChunk
{
    std::string content;
    DocumentChunkMetadata metadata;
    std::optional<Embedding> embedding;
};

struct RetrievedDocumentChunk
{
    std::string content;
    uint64_t chunk_id;
    float score;
};

class IDocumentChunkRetriever
{
  public:
    virtual ~IDocumentChunkRetriever() = default;

    virtual std::vector<RetrievedDocumentChunk> retrieve(const std::string& question, const size_t top_k) const = 0;
    virtual void add_document_chunks(const std::vector<DocumentChunk>& chunks) = 0;
    virtual void dump(std::ostream& output) const = 0;
    virtual void load(std::istream& input) = 0;
};

std::unique_ptr<IDocumentChunkRetriever> create_document_chunk_retriever(const DocumentChunkRetrieverParams& params);

class SimpleDocumentChunkRetriever : public IDocumentChunkRetriever
{
  public:
    explicit SimpleDocumentChunkRetriever(std::unique_ptr<IEmbeddingCalculator>&& embedding_calculator,
                                          std::unique_ptr<IVectorStore>&& vector_store);

    std::vector<RetrievedDocumentChunk> retrieve(const std::string& question, const size_t top_k) const override;
    void add_document_chunks(const std::vector<DocumentChunk>& chunks) override;
    void dump(std::ostream& output) const override;
    void load(std::istream& input) override;

  private:
    std::unique_ptr<IEmbeddingCalculator> embedding_calculator_;
    std::unique_ptr<IVectorStore> vector_store_;
    std::vector<DocumentChunk> document_chunks_;
};
} // namespace ds