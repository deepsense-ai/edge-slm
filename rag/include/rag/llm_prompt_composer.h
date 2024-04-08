#pragma once

#include <string>
#include <memory>
#include <filesystem>
#include "rag/document_retrieval.h"

namespace ds {
    class ILLMPromptComposer {
        public:
        virtual ~ILLMPromptComposer() = default;

        virtual std::string create(const std::string& user_query, const std::vector<RetrievedDocumentChunk>& document_contexts) const = 0;
    };

    std::unique_ptr<ILLMPromptComposer> create_llm_prompt_composer(const std::filesystem::path& template_path);
}