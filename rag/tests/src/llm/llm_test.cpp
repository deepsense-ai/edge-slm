#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <ostream>
#include <sstream>
#include <thread>

#include "llm/llama_provider.h"

namespace ds
{

std::istringstream get_default_config_iss()
{
    const std::string config_str = "{}";
    std::istringstream iss{config_str};

    return iss;
}

std::filesystem::path get_default_model_path()
{
    const auto env_val = std::getenv("LLM_TEST_MODEL_PATH");

    return env_val ? env_val : "./model.ggml";
}

const auto default_model_path = get_default_model_path();

class LlamaProviderTest : public ::testing::Test
{
};

TEST_F(LlamaProviderTest, normalGeneration)
{
    auto iss = get_default_config_iss();
    auto llm = LlamaProvider::from_config(default_model_path, iss);
    const LlmInput input = {.prompt = "Wnen does summer start?"};
    const auto result = llm->generate(input);

    EXPECT_FALSE(result.answer.empty());
}

TEST_F(LlamaProviderTest, asyncGeneration)
{
    auto iss = get_default_config_iss();
    std::unique_ptr<ILlmProvider> llm = LlamaProvider::from_config(default_model_path, iss);

    const LlmInput input = {.prompt = "Wnen does summer start?"};
    std::string joined_answer;
    AsyncStatus status = AsyncStatus::GENERATING;

    auto callback = [&joined_answer, &status](const LlmAsyncOutput& output)
    {
        joined_answer += output.answer;
        status = output.status;
    };

    auto textGeneration = llm->generateAsync(input, callback);
    auto generationFuture = textGeneration->start();
    auto finalResult = generationFuture.get();

    ASSERT_EQ(status, AsyncStatus::FINISHED);
    EXPECT_FALSE(joined_answer.empty());
    EXPECT_EQ(joined_answer, finalResult.answer);
}

TEST_F(LlamaProviderTest, asyncGenerationMaxTokens)
{
    const std::string config_str = "{\"max_tokens\": 5}";
    std::istringstream iss{config_str};

    std::unique_ptr<ILlmProvider> llm = LlamaProvider::from_config(default_model_path, iss);

    const LlmInput input = {.prompt = "What does PC stand for?"};
    std::string joined_answer;
    uint32_t callback_counter = 0;
    AsyncStatus status = AsyncStatus::GENERATING;

    auto callback = [&joined_answer, &status, &callback_counter](const LlmAsyncOutput& output)
    {
        joined_answer += output.answer;
        status = output.status;
        ++callback_counter;
    };

    auto textGeneration = llm->generateAsync(input, callback);
    auto generationFuture = textGeneration->start();
    auto finalResult = generationFuture.get();

    ASSERT_EQ(status, AsyncStatus::FINISHED);
    EXPECT_FALSE(joined_answer.empty());
    EXPECT_EQ(joined_answer, finalResult.answer);
    EXPECT_EQ(callback_counter, 5);
}

TEST_F(LlamaProviderTest, asyncGenerationMultiplePromptsClearContext)
{
    auto iss = get_default_config_iss();
    std::unique_ptr<ILlmProvider> llm = LlamaProvider::from_config(default_model_path, iss);

    const LlmInput firstInput = {.prompt = "Answer in one sentence. Wnen does summer start?"};
    std::string joined_answer;
    AsyncStatus status = AsyncStatus::GENERATING;

    auto callback = [&joined_answer, &status](const LlmAsyncOutput& output)
    {
        joined_answer += output.answer;
        status = output.status;
    };

    auto firstTextGeneration = llm->generateAsync(firstInput, callback);
    auto firstGenerationFuture = firstTextGeneration->start();
    auto firstResult = firstGenerationFuture.get();

    ASSERT_EQ(status, AsyncStatus::FINISHED);
    ASSERT_FALSE(joined_answer.empty());
    ASSERT_EQ(joined_answer, firstResult.answer);

    llm->clear_context();

    const LlmInput secondInput = {.prompt = "Answer in one sentence. When does winter start?"};
    joined_answer.clear();
    status = AsyncStatus::GENERATING;

    auto secondTextGeneration = llm->generateAsync(secondInput, callback);
    auto secondGenerationFuture = secondTextGeneration->start();
    auto secondResult = secondGenerationFuture.get();

    ASSERT_EQ(status, AsyncStatus::FINISHED);
    EXPECT_FALSE(joined_answer.empty());
    EXPECT_EQ(joined_answer, secondResult.answer);
}

TEST_F(LlamaProviderTest, asyncGenerationForceStop)
{
    auto iss = get_default_config_iss();
    std::unique_ptr<ILlmProvider> llm = LlamaProvider::from_config(default_model_path, iss);

    const LlmInput input = {.prompt = "Wnen does summer start?"};
    std::string joined_answer;
    AsyncStatus status = AsyncStatus::GENERATING;

    auto callback = [&joined_answer, &status](const LlmAsyncOutput& output)
    {
        joined_answer += output.answer;
        status = output.status;
    };

    auto textGeneration = llm->generateAsync(input, callback);
    auto generationFuture = textGeneration->start();

    ASSERT_TRUE(generationFuture.valid());

    textGeneration->stop();

    auto finalResult = generationFuture.get();

    ASSERT_EQ(status, AsyncStatus::STOPPED);
    EXPECT_FALSE(joined_answer.empty());
    EXPECT_EQ(joined_answer, finalResult.answer);
}

TEST_F(LlamaProviderTest, normalGenerationManyPrompts)
{
    // This scenario tests whether several generations are valid
    // Checks if there's no segmentation fault or exception is thrown

    auto iss = get_default_config_iss();
    std::unique_ptr<ILlmProvider> llm = LlamaProvider::from_config(default_model_path, iss);

    static const std::array<LlmInput, 2> inputs = {
        {{.prompt = "Who is Ada Lovelace?"}, {.prompt = "When was she born?"}}};

    for(const auto& input : inputs)
    {
        const auto result = llm->generate(input);

        EXPECT_FALSE(result.answer.empty());
    }

    SUCCEED();
}

TEST_F(LlamaProviderTest, parsingJson)
{
    LlamaParameters params;

    const std::string config_str =
        "{\"temp\": 0.1, \"max_predict_tokens\": 128, \"context_size\": 4096, \"threads\": 4, \"seed\": 1}";
    std::istringstream iss{config_str};

    EXPECT_NO_THROW(LlamaProvider::from_config(default_model_path, iss););
}

} // namespace ds
