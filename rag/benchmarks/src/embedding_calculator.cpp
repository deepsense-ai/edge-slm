#include "rag/embedding_calculator.h"
#include <benchmark/benchmark.h>

#include "mem_usage.h"
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <ranges>

namespace ds
{

std::string get_embedding_model_path_from_env()
{
    const auto val = std::getenv("EMBEDDING_MODEL_PATH");
    return val ? val : "./gte-base-f32.gguf";
}

int32_t get_batch_size_from_env()
{
    const auto val = std::getenv("BATCH_SIZE");
    return val ? std::stoi(val) : 512;
}

int32_t get_repetitions_from_env()
{
    const auto val = std::getenv("REPETITIONS");
    return val ? std::stoi(val) : 1;
}

const std::array<std::string, 3>
    TEXTS_TO_EMBED({
        "Hello World!", // 5 tokens
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo "
        "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id "
        "est "
        "laborum.", // 160 tokens
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
        "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
        "nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
        "anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque "
        "laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae "
        "dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia "
        "consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem "
        "ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut "
        "labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem "
        "ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure "
        "reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum "
        "fugiat quo voluptas nulla pariatur? Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut" // 512 tok
    });

static void VectorEmbeddings(benchmark::State& state)
{
    using namespace std::chrono;
    using namespace std::chrono_literals;

    const auto batch_size = get_batch_size_from_env();
    const auto model_path = get_embedding_model_path_from_env();
    const auto repetitions = get_repetitions_from_env();
    const auto texts = std::vector<std::string>(repetitions, TEXTS_TO_EMBED[state.range(0)]);
    const int n_threads = state.range(1);

    const auto embedding_calculator = embedding_calculator_factory(
        EmbeddingCalculatorParams{.model_path = model_path, .n_threads = n_threads, .batch_size = batch_size});
    uint64_t total_tokens = 0;

    const auto start = high_resolution_clock::now();
    for(auto _ : state)
    {
        const auto embedding_results = embedding_calculator->calc_batch(texts);
        total_tokens += std::accumulate(embedding_results.begin(), embedding_results.end(), 0,
                                        [](uint64_t current_total, const EmbeddingCalculationResult& result)
                                        { return current_total + result.n_tokens; });
    }
    const auto end = high_resolution_clock::now();
    const auto duration = (end - start) / 1.0s;

    state.counters["sum_tokens"] = total_tokens;
    state.counters["avg_tokens_per_second"] = total_tokens / duration;
}

static void MemSummary(benchmark::State& state)
{
    // A dummy function to put the result of the peak memory usage
    for(auto _ : state)
    {
    }
    state.counters["memory_gb"] = get_peak_process_mem_usage_gb();
}

BENCHMARK(VectorEmbeddings)
    ->ArgNames({"text_id", "n_threads"})
    ->ArgsProduct({{0, 1, 2}, {1, 4, 8, 16}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(MemSummary);

BENCHMARK_MAIN();

} // namespace ds