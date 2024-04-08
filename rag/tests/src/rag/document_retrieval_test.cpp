#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rag/document_retrieval.h"
#include "test_utils.h"

namespace ds
{

class DocumentRetrievalTest : public ::testing::Test
{
  public:
    const std::filesystem::path MODEL_PATH = ASSETS_ROOT / "embeddings/gte-base/gte-base-f32.gguf";
};

TEST_F(DocumentRetrievalTest, CheckCreation)
{
    auto document_retriever = create_document_chunk_retriever(DocumentChunkRetrieverParams{
        EmbeddingCalculatorParams{.model_path = MODEL_PATH, .n_threads = 1, .batch_size = 512}});

    EXPECT_NE(document_retriever, nullptr);
}

TEST_F(DocumentRetrievalTest, CheckSimpleCreate3Sentences)
{ // Expected results based on: https://huggingface.co/thenlper/gte-large
    auto document_retriever = create_document_chunk_retriever(DocumentChunkRetrieverParams{
        EmbeddingCalculatorParams{.model_path = MODEL_PATH, .n_threads = 1, .batch_size = 512}});

    document_retriever->add_document_chunks({DocumentChunk{"That is a happy dog"},
                                             DocumentChunk{"That is a very happy person"},
                                             DocumentChunk{"Today is a sunny day"}});

    const auto result = document_retriever->retrieve("That is a happy person", 2);
    EXPECT_EQ(result[0].content, "That is a very happy person");
    EXPECT_EQ(result[1].content, "That is a happy dog");
}

TEST_F(DocumentRetrievalTest, CheckLoadingDataWithPrecalculatedEmbeddings)
{
    class MockEmbeddingCalculator : public IEmbeddingCalculator
    {
      public:
        MOCK_METHOD(EmbeddingCalculationResult, calc, (const std::string& chunk), (const override));
        MOCK_METHOD(std::vector<EmbeddingCalculationResult>, calc_batch, (const std::vector<std::string>& chunks),
                    (const override));

        MOCK_METHOD(size_t, get_embedding_rank, (), (const override));
    };

    auto embedding_calculator = std::make_unique<MockEmbeddingCalculator>();
    EXPECT_CALL(*embedding_calculator, get_embedding_rank()).WillRepeatedly(::testing::Return(3));
    EXPECT_CALL(*embedding_calculator, calc("Query"))
        .WillRepeatedly(::testing::Return(EmbeddingCalculationResult{.embedding = {0.6f, 0.1f, 0.f}}));
    EXPECT_CALL(*embedding_calculator, calc_batch(std::vector<std::string>({})));

    auto document_retriever = SimpleDocumentChunkRetriever(std::move(embedding_calculator), vector_store_factory(3));

    document_retriever.add_document_chunks(
        {DocumentChunk{"Chunk_1", DocumentChunkMetadata{}, Embedding{1.0f, 0.f, 0.f}},
         DocumentChunk{"Chunk_2", DocumentChunkMetadata{}, Embedding{0.f, 1.f, 0.f}},
         DocumentChunk{"Chunk_3", DocumentChunkMetadata{}, Embedding{0.f, 1.f, 0.f}}});

    const auto result = document_retriever.retrieve("Query", 1);
    EXPECT_EQ(result[0].content, "Chunk_1");

    const auto result_2 = document_retriever.retrieve("Query", 2);
    EXPECT_EQ(result_2[0].content, "Chunk_1");
    EXPECT_EQ(result_2[1].content, "Chunk_2");
}

} // namespace ds