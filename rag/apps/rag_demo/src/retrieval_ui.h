#pragma once

#include <fmt/format.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <tabulate/table.hpp>
#include <nlohmann/json.hpp>
#include "result.h"
#include "options.h"
#include "rag/document_retrieval.h"

namespace ds
{

using RetrievalFn = std::vector<RetrievedDocumentChunk>(const std::string&);

void run_retrieval_interactive(const std::function<RetrievalFn>& retrieval_fn)
{

    std::string query;

    while(true)
    {
        std::cout << "\n\nYour query: ";
        std::getline(std::cin, query);
        if(query.empty())
            break;
        const auto most_relevant = with_time_report("Document retrieval", retrieval_fn, query);

        using namespace tabulate;
        Table responses;

        responses.add_row({"Document chunk", "Score", "Id"});
        responses.column(0).format().width(100);
        responses.column(1).format().width(10);
        responses.column(2).format().width(5);

        std::ranges::for_each(
            most_relevant,
            [&responses](const RetrievedDocumentChunk& chunk)
            {
                const auto content = boost::replace_all_copy(chunk.content, "\n", " ");
                responses.add_row({content, fmt::format("{:.4f}", chunk.score), std::to_string(chunk.chunk_id)});
            });
        responses.row(0).format().font_style({FontStyle::bold});
        std::cout << responses << std::endl;
    }
}

void run_retrieval_non_interactive(const std::function<RetrievalFn>& retrieval_fn, const Options& options)
{

    std::string query;
    std::vector<RetrievalResult> results;

    std::ifstream file(options.queries_input, std::ios::in);

    if(!file.is_open())
        throw std::runtime_error(fmt::format("Could not open the input file [{}].", options.queries_input));

    while(std::getline(file, query))
    {
        const auto most_relevant = with_time_report("Document retrieval", retrieval_fn, query);
        std::vector<std::string> chunk_contents;
        std::ranges::transform(most_relevant, std::back_inserter(chunk_contents),
                               [](const RetrievedDocumentChunk& doc) { return doc.content; });
        results.push_back(RetrievalResult{query, chunk_contents});
    }

    std::ofstream output_file(options.result_output, std::ios::out);
    if(!output_file.is_open())
        throw std::runtime_error(fmt::format("Could not open the output file [{}].", options.result_output));
    output_file << nlohmann::json(results);
}

void run_retrieval_mode(std::shared_ptr<IDocumentChunkRetriever> retriever, const Options& options)
{
    const auto document_retrieval_fn = [&retriever, &options](const std::string& query)
    { return retriever->retrieve(query, options.top_k); };
    if(!options.queries_input.empty())
    {
        run_retrieval_non_interactive(document_retrieval_fn, options);
    }
    else
    {
        run_retrieval_interactive(document_retrieval_fn);
    }
}
} // namespace ds