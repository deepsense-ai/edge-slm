#include "llm/llama_provider.h"
#include <benchmark/benchmark.h>

#include "mem_usage.h"
#include <iostream>
#include <random>
#include <ranges>

namespace ds
{

std::filesystem::path get_default_model_path()
{
    const auto env_val = std::getenv("LLM_TEST_MODEL_PATH");

    return env_val ? env_val : "./model.ggml";
}

uint32_t get_default_model_max_tokens()
{
    const auto env_val = std::getenv("LLM_TEST_MAX_TOKENS");

    return env_val ? std::stoul(env_val) : 5;
}

uint32_t get_default_model_context_window()
{
    const auto env_val = std::getenv("LLM_TEST_CONTEXT_WINDOW");

    return env_val ? std::stoul(env_val) : 0;
}

const auto default_model_path = get_default_model_path();
const auto default_max_tokens = get_default_model_max_tokens();
const auto default_context_window = get_default_model_context_window();

const std::array<std::string, 1> PROMPTS{
    "What does PC stand for?", // 8 tokens
};

static void LLMInference(benchmark::State& state)
{
    const uint32_t n_threads = state.range(0);

    const auto params = LlamaParameters{
        .temp = 0.8f, .max_tokens = default_max_tokens, .context_size = default_context_window, .threads = n_threads};
    std::unique_ptr<ILlmProvider> llm = std::make_unique<LlamaProvider>(default_model_path, params);

    const LlmInput input = {.prompt = PROMPTS[0]};

    std::vector<LlmTimings> timings;
    for(auto _ : state)
    {
        auto response = llm->generate(input);
        timings.push_back(response.timings);
    }

    LlmTimings timings_sum = std::accumulate(
        timings.begin(), timings.end(), LlmTimings(),
        [](const LlmTimings& total, const LlmTimings& current)
        {
            return LlmTimings{
                .tokens_per_second = total.tokens_per_second + current.tokens_per_second,
                .generation_time_seconds = total.generation_time_seconds + current.generation_time_seconds,
                .tokenization_time_seconds = total.tokenization_time_seconds + current.tokenization_time_seconds,
                .prompt_decoding_time_seconds = total.prompt_decoding_time_seconds + current.prompt_decoding_time_seconds};
        });

    state.counters["tok/s"] = timings_sum.tokens_per_second / timings.size();
    state.counters["tokenization"] = timings_sum.tokenization_time_seconds / timings.size();
    state.counters["prompt_decoding"] = timings_sum.prompt_decoding_time_seconds / timings.size();
}

static void MemSummary(benchmark::State& state)
{
    // A dummy function to put the result of the peak memory usage
    for(auto _ : state)
    {
    }
    state.counters["memory_gb"] = get_peak_process_mem_usage_gb();
}

BENCHMARK(LLMInference)
    ->ArgNames({"n_threads"})
    ->ArgsProduct({{1, 2, 3, 4, 5, 6, 7}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(MemSummary); // That's a hack. We must call this in order to memory readout to take place.

BENCHMARK_MAIN();
} // namespace ds