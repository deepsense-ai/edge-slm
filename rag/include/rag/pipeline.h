#pragma once

#include "llm/llm_provider.h"
#include "rag/document_retrieval.h"
#include "rag/llm_prompt_composer.h"

#include <cstdint>
#include <functional>
#include <memory>

namespace ds
{

struct RagInferenceSettings
{
    std::uint32_t top_k;
};

class RagPipeline
{
  public:
    RagPipeline(std::shared_ptr<IDocumentChunkRetriever> document_retriever,
                std::unique_ptr<ILLMPromptComposer>&& prompt_composer, std::unique_ptr<ILlmProvider>&& llm_provider)
        : document_retriever_(document_retriever), prompt_composer_(std::move(prompt_composer)),
          llm_provider_(std::move(llm_provider))
    {
    }

    LlmOutput generate(const std::string& query, const RagInferenceSettings& settings);
    std::unique_ptr<ILlmAsyncGeneration> generate_async(const std::string& query, const RagInferenceSettings& settings,
                                                        const std::function<LlmCallback>& callback);
    void clear_chat_context();

  private:
    LlmInput prepare_inference_prompt_(const std::string& user_input, const RagInferenceSettings& settings);

    std::shared_ptr<IDocumentChunkRetriever> document_retriever_;
    std::unique_ptr<ILLMPromptComposer> prompt_composer_;
    std::unique_ptr<ILlmProvider> llm_provider_;
};
} // namespace ds
