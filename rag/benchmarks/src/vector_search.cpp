#include "rag/vector_database.h"
#include <benchmark/benchmark.h>

#include <random>
#include <iostream>
#include "mem_usage.h"

namespace ds
{

std::vector<Embedding> create(size_t embedding_rank, size_t N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<Embedding> chunks(N);

    for(size_t i = 0; i < N; i++)
    {
        chunks[i] = Embedding{std::vector<float>(embedding_rank)};
        for(size_t j = 0; j < embedding_rank; j++)
        {
            chunks[i][j] = dist(gen);
        }
    }

    return chunks;
}

static void IndexingVector(benchmark::State& state)
{
    const size_t embedding_rank = state.range(0);
    const size_t items = state.range(1);

    auto chunks = create(embedding_rank, items);

    for(auto _ : state)
    {
        auto db = vector_store_factory(embedding_rank);
        db->add(chunks);
    }
}

static void DatabaseLookup(benchmark::State& state)
{
    const size_t embedding_rank = state.range(0);
    const size_t items = state.range(1);
    const size_t top_k = state.range(1);

    auto db = vector_store_factory(embedding_rank);
    auto chunks = create(embedding_rank, items);

    db->add(chunks);
    const std::vector<float> search(embedding_rank);

    for(auto _ : state)
    {
        auto result = db->retrieve(search, top_k);
        benchmark::DoNotOptimize(result);
    }
}

static void MemSummary(benchmark::State& state) {
    // A dummy function to put the result of the peak memory usage
    for(auto _ : state) {}
    state.counters["memory_gb"] = get_peak_process_mem_usage_gb();
}


BENCHMARK(IndexingVector)
    ->ArgNames({"embedding_rank", "n_elements"})
    ->ArgsProduct({{512, 768, 1024, 4096}, {100, 1000, 10000, 30000}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(DatabaseLookup)
    ->ArgNames({"embedding_rank", "n_elements", "top_k"})
    ->ArgsProduct({{512, 768, 1024, 4096}, {100, 1000, 10000, 30000}, {1, 5, 10}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(MemSummary); // That's a hack. We must call this in order to memory readout to take place.


BENCHMARK_MAIN();
} // namespace ds