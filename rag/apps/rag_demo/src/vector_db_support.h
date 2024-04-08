#pragma once

#include "llm/utils.h"
#include "options.h"
#include "rag/document_retrieval.h"
#include <fstream>

namespace ds
{

std::shared_ptr<IDocumentChunkRetriever> prepare_doc_chunk_retriever(const Options& options)
{
    auto retriever = create_document_chunk_retriever(DocumentChunkRetrieverParams{
        .embedding_calculator_params = EmbeddingCalculatorParams{.model_path = options.embedding_model_path,
                                                                 .n_threads = options.embedding_threads,
                                                                 .batch_size = options.embedding_batch_size}});

    if(!options.database_input.empty())
    {
        std::ifstream input_file(options.database_input, std::ios::in);
        auto load_db_fn = [&retriever, &input_file]() { retriever->load(input_file); };
        with_time_report("Document DB load", load_db_fn);
    }

    return retriever;
}

void dump_vector_db_if_requested(const IDocumentChunkRetriever& retriever, Options& options)
{
    if(!options.database_output.empty())
    {
        std::ofstream output_file(options.database_output, std::ios::out);
        retriever.dump(output_file);
    }
}
} // namespace ds