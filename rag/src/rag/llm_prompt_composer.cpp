#include "rag/llm_prompt_composer.h"
#include "llm/utils.h"

#include <boost/algorithm/string.hpp>
#include <kainjow/mustache.hpp>

#include <fstream>
#include <ranges>

namespace ds
{
using namespace kainjow::mustache;

class MustacheLLMPromptComposer : public ILLMPromptComposer
{
  public:
    explicit MustacheLLMPromptComposer(const std::string& templ) : template_(templ) {}

    std::string create(const std::string& user_query,
                       const std::vector<RetrievedDocumentChunk>& document_contexts) const override
    {
        std::vector<std::string> contexts_vec;
        std::ranges::transform(document_contexts, std::back_inserter(contexts_vec),
                       [](const auto& doc_chunk) { return doc_chunk.content; });

        auto contexts = boost::algorithm::join(contexts_vec, " ");

        data data;
        data.set("query", {user_query});
        data.set("context", {contexts});

        return mustache(template_).render(data);
    }

  private:
    std::string template_;
};

std::unique_ptr<ILLMPromptComposer> create_llm_prompt_composer(const std::filesystem::path& template_path) {
    std::ifstream file_stream(template_path, std::ios::in);
    std::stringstream buffer;
    buffer << file_stream.rdbuf();

    return std::make_unique<MustacheLLMPromptComposer>(buffer.str());
}


} // namespace ds