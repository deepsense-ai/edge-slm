#pragma once

#include "llm/llama_provider.h"
#include "options.h"
#include "rag/llm_prompt_composer.h"
#include "rag/pipeline.h"

#include <spdlog/spdlog.h>

namespace ds
{

void run_chat_interactive_mode(RagPipeline& pipeline, const Options& options)
{
    std::string query;
    while(true)
    {
        std::cout << "\nYour query: ";
        std::getline(std::cin, query);
        if(query.empty())
            break;

        std::cout << "\nResponse:" << std::endl;
        auto async_generation = pipeline.generate_async(query, RagInferenceSettings{.top_k = options.top_k},
                                                        [](const LlmAsyncOutput& output)
                                                        {
                                                            std::cout << output.answer;
                                                            std::cout.flush();
                                                        });

        auto generation_future = async_generation->start();
        auto final_result = generation_future.get();

        pipeline.clear_chat_context();

        std::cout << "\n\n=====ANSWER SUMMARY=====\n";
        std::cout << fmt::format("Time to first token: {:.2f} seconds.\n", final_result.timings.prompt_decoding_time_seconds);
        std::cout << fmt::format("LLM response generation speed: {:.2f} tok/s.\n", final_result.timings.tokens_per_second);
        std::cout << "=====END OF SUMMARY=====" << std::endl;;
    }
}

void run_chat_mode(std::shared_ptr<IDocumentChunkRetriever> retriever, const Options& options)
{
    if(options.model_path.empty())
        throw std::invalid_argument("Model path must be provided in chat mode.");
    if(options.model_config_path.empty())
        throw std::invalid_argument("Model config path must be provided in chat mode.");
    if(options.prompt_template_path.empty())
        throw std::invalid_argument("Prompt template path must be provided in chat mode.");

    auto rag_pipeline = RagPipeline(retriever, create_llm_prompt_composer(options.prompt_template_path),
                                    LlamaProvider::from_config(options.model_path, options.model_config_path));

    if(!options.queries_input.empty())
    {
        throw std::logic_error("Chat non interactive mode is not yet supported.");
    }

    run_chat_interactive_mode(rag_pipeline, options);
}

} // namespace ds
