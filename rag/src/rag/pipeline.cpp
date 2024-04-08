#include "rag/pipeline.h"
#include <spdlog/spdlog.h>
namespace ds
{

LlmOutput RagPipeline::generate(const std::string& query, const RagInferenceSettings& settings)
{
    const auto inference_prompt = prepare_inference_prompt_(query, settings);

    return llm_provider_->generate(inference_prompt);
}

std::unique_ptr<ILlmAsyncGeneration> RagPipeline::generate_async(const std::string& query,
                                                                 const RagInferenceSettings& settings,
                                                                 const std::function<LlmCallback>& callback)
{
    const auto inference_prompt = prepare_inference_prompt_(query, settings);

    spdlog::debug("Query: {}", query, inference_prompt.prompt);
    spdlog::debug("Prompt:\n{}", inference_prompt.prompt);

    return llm_provider_->generateAsync(inference_prompt, callback);
}

LlmInput RagPipeline::prepare_inference_prompt_(const std::string& query, const RagInferenceSettings& settings)
{
    const auto contexts = document_retriever_->retrieve(query, settings.top_k);
    const auto prompt = prompt_composer_->create(query, contexts);

    return LlmInput{.prompt = prompt};
}

void RagPipeline::clear_chat_context()
{
    llm_provider_->clear_context();
}
} // namespace ds
