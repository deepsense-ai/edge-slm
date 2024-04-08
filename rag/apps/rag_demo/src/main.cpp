#include <algorithm>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "chat_ui.h"
#include "llm/llama_provider.h"
#include "llm/utils.h"
#include "options.h"
#include "retrieval_ui.h"
#include "vector_db_support.h"

int main(int argc, char* argv[])
{
    using namespace ds;

    spdlog::set_level(spdlog::level::debug);

    auto options = OptionParser::parse_options(argc, argv);

    auto document_chunk_retriever = prepare_doc_chunk_retriever(options);
    dump_vector_db_if_requested(*document_chunk_retriever, options);

    switch(options.mode)
    {
        case ProgramMode::RETRIEVAL:
            run_retrieval_mode(document_chunk_retriever, options);
            break;
        case ProgramMode::CHAT:
            run_chat_mode(document_chunk_retriever, options);
            break;
    }

    return 0;
}