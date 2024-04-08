#include <gtest/gtest.h>

#include "rag/llm_prompt_composer.h"
#include "test_utils.h"

namespace ds
{

class LLMPromptComposer : public ::testing::Test
{
    public:
    const std::filesystem::path PROMPT_DIR = ASSETS_ROOT / "prompt_templates";
};

TEST_F(LLMPromptComposer, CheckPhi2SimplePrompt)
{
    const auto prompt_composer = create_llm_prompt_composer(PROMPT_DIR / "phi_2.mustache");

    const auto expected = "Instruct: Generate short, concise answer to the User Question, using provided Contexts. Make sure you use the Contexts, which contain relevant information for constructing the answer. Note that not all the contexts must be relevant. Read carefully and use reasoning. Contexts might be not visible to the user, so please give short answer by quoting relevant part of the context(s) if possible. Contexts are provided as follows:"
                          "\nContexts: content chunk 1."
                          "\nQuestion: user_query"
                          "\nOutput:";

    const auto prompt =
        prompt_composer->create("user_query", {RetrievedDocumentChunk{.content = "content chunk 1.", .score = 1.0f}});
    EXPECT_EQ(prompt, expected);
}

TEST_F(LLMPromptComposer, CheckPhi2PromptWithMultipleContexts)
{
    const auto prompt_composer = create_llm_prompt_composer(PROMPT_DIR / "phi_2.mustache");

    const auto expected = "Instruct: Generate short, concise answer to the User Question, using provided Contexts. Make sure you use the Contexts, which contain relevant information for constructing the answer. Note that not all the contexts must be relevant. Read carefully and use reasoning. Contexts might be not visible to the user, so please give short answer by quoting relevant part of the context(s) if possible. Contexts are provided as follows:"
                          "\nContexts: content chunk 1. content chunk 2."
                          "\nQuestion: user_query"
                          "\nOutput:";

    const auto prompt =
        prompt_composer->create("user_query", {RetrievedDocumentChunk{.content = "content chunk 1.", .score = 1.0f},
                                               RetrievedDocumentChunk{.content = "content chunk 2.", .score = 1.0f}});
    EXPECT_EQ(prompt, expected);
}

TEST_F(LLMPromptComposer, CheckLLama1_1BSimplePrompt)
{
    const auto prompt_composer = create_llm_prompt_composer(PROMPT_DIR / "tiny_llama1.1B.mustache");

    const auto expected = "<|system|>"
                          "\nGenerate short, concise answer to the Question, using provided context."
                          "\nContext: content chunk 1.</s>"
                          "\n<|user|>"
                          "\nQuestion: user_query</s>";

    const auto prompt =
        prompt_composer->create("user_query", {RetrievedDocumentChunk{.content = "content chunk 1.", .score = 1.0f}});
    EXPECT_EQ(prompt, expected);
}

} // namespace ds