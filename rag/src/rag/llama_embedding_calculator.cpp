#include "llm/utils.h"
#include "rag/embedding_calculator.h"

#include "llamacpp/common.h"
#include "llamacpp/llama.h"

#include <algorithm>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <vector>

namespace ds
{
template <typename T>
using unique_ptr_with_deleter = std::unique_ptr<T, std::function<void(T*)>>;

using LlamaCtxUniquePtr = unique_ptr_with_deleter<llama_context>;

class LlamaBackendManager
{
  public:
    static LlamaBackendManager& get_instance()
    {
        static LlamaBackendManager instance;
        return instance;
    }

    LlamaBackendManager(const LlamaBackendManager&) = delete;
    LlamaBackendManager& operator=(const LlamaBackendManager&) = delete;
    LlamaBackendManager(LlamaBackendManager&&) = delete;
    LlamaBackendManager& operator=(LlamaBackendManager&&) = delete;
    ~LlamaBackendManager() { llama_backend_free(); }

  private:
    LlamaBackendManager() { llama_backend_init(); }
};

static void normalize(std::span<const float> vec, std::span<float> out)
{
    float norm = std::accumulate(vec.begin(), vec.end(), 0.f,
                                 [](const float& sum, const float& curr) { return sum + curr * curr; });
    norm = std::sqrt(norm);
    std::ranges::transform(vec, out.begin(), [&norm](const float& val) { return val / norm; });
}

static void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, int seq_id)
{
    for(size_t i = 0; i < tokens.size(); i++)
    {
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

using TokenizedSequence = std::vector<std::int32_t>;

class LLamaEmbeddingCalculator : public IEmbeddingCalculator
{
  public:
    explicit LLamaEmbeddingCalculator(const EmbeddingCalculatorParams& params) : max_batch_(params.batch_size)
    {
        gpt_params_.embedding = true;
        gpt_params_.model = params.model_path;
        gpt_params_.n_threads = params.n_threads;
        gpt_params_.n_batch = max_batch_;

        LlamaBackendManager::get_instance();

        std::tie(model, ctx) = llama_init_from_gpt_params(gpt_params_);

        if(model == nullptr || ctx == nullptr)
        {
            free_llama_pointers();
            throw std::runtime_error(fmt::format("Could not load model from file: {}", params.model_path.c_str()));
        }

        const auto n_ctx = llama_n_ctx(ctx);
        if(max_batch_ < n_ctx)
        {
            free_llama_pointers();
            throw std::runtime_error(
                fmt::format("Cannot create embeddings model. The Batch size is smaller than context. Got: {} and {}",
                            max_batch_, n_ctx));
        }

        embedding_rank_ = llama_n_embd(model);
    }

    ~LLamaEmbeddingCalculator() { free_llama_pointers(); }

    std::vector<EmbeddingCalculationResult> calc_batch(const std::vector<std::string>& chunks) const override;

    size_t get_embedding_rank() const override { return embedding_rank_; }

  private:
    gpt_params gpt_params_;
    llama_model* model;
    llama_context* ctx;

    size_t embedding_rank_;
    size_t max_batch_;

    std::vector<TokenizedSequence> tokenized_sequences_(const std::vector<std::string>& chunks) const;
    void batch_decode_(llama_batch& batch, std::span<float> output) const;
    void copy_and_normalize_embeddings_(llama_batch& batch, std::span<float> output) const;
    std::optional<std::span<const float>> get_raw_embedding_(llama_batch& batch, int batch_idx) const;
    std::vector<float> calc_in_fitting_batches_(const std::vector<TokenizedSequence>& tokenized_sequences) const;

    void free_llama_pointers();
};

void LLamaEmbeddingCalculator::batch_decode_(llama_batch& batch, std::span<float> output) const
{
    llama_kv_cache_clear(ctx);

    if(auto ret_code = llama_decode(ctx, batch); ret_code < 0)
    {
        spdlog::error("Embedding calculation failed, decode error code: {}", ret_code);
        throw std::runtime_error("Embedding calculation failed - llama_decode.");
    }

    copy_and_normalize_embeddings_(batch, output);
}

void LLamaEmbeddingCalculator::copy_and_normalize_embeddings_(llama_batch& batch, std::span<float> output) const
{
    for(int i = 0; i < batch.n_tokens; i++)
    {
        if(!batch.logits[i])
        {
            continue;
        }
        const auto batch_seq_id = batch.seq_id[i][0];
        auto raw_embedding = get_raw_embedding_(batch, i);
        if(!raw_embedding)
        {
            throw std::runtime_error("Could not retrieve raw embeddings buffer.");
        }

        auto normalized_embedding = output.subspan(batch_seq_id * embedding_rank_, embedding_rank_);
        normalize(*raw_embedding, normalized_embedding);
    }
}

std::vector<TokenizedSequence>
LLamaEmbeddingCalculator::tokenized_sequences_(const std::vector<std::string>& sequences) const
{
    std::vector<TokenizedSequence> tokenized_sequences;
    tokenized_sequences.reserve(sequences.size());

    std::ranges::transform(sequences, std::back_inserter(tokenized_sequences),
                           [this](const std::string& sequence)
                           {
                               auto tokens = ::llama_tokenize(ctx, sequence, true);
                               if(tokens.size() > max_batch_)
                                   tokens.resize(max_batch_);
                               return tokens;
                           });

    return tokenized_sequences;
}

std::optional<std::span<const float>> LLamaEmbeddingCalculator::get_raw_embedding_(llama_batch& batch,
                                                                                   int batch_idx) const
{
    // the whole fallback mechanism was adapted directly from llama.cpp examples.
    auto embedding_ptr = llama_get_embeddings_seq(ctx, batch.seq_id[batch_idx][0]);
    if(embedding_ptr == nullptr)
    {
        embedding_ptr = llama_get_embeddings_ith(ctx, batch_idx);
        if(embedding_ptr == nullptr)
        {
            spdlog::error("Failed to get embeddings for batch_id: {}", batch_idx);
            return std::nullopt;
        }
    }

    return std::span<const float>(embedding_ptr, embedding_rank_);
}

std::vector<EmbeddingCalculationResult>
LLamaEmbeddingCalculator::calc_batch(const std::vector<std::string>& sequences) const
{
    const auto n_sequences = sequences.size();
    if (n_sequences == 0)
        return {};

    const auto tokenized_sequences = tokenized_sequences_(sequences);
    const auto embeddings = calc_in_fitting_batches_(tokenized_sequences);

    std::vector<EmbeddingCalculationResult> result;
    result.reserve(n_sequences);

    for(size_t i = 0; i < n_sequences; i++)
    {
        result.emplace_back(
            EmbeddingCalculationResult{.embedding = Embedding(embeddings.begin() + i * embedding_rank_,
                                                              embeddings.begin() + (i + 1) * embedding_rank_),
                                       .n_tokens = tokenized_sequences[i].size()});
    }

    return result;
}

std::vector<float>
LLamaEmbeddingCalculator::calc_in_fitting_batches_(const std::vector<TokenizedSequence>& tokenized_sequences) const
{
    const int n_sequences = tokenized_sequences.size();
    auto batch = llama_batch_init(max_batch_, 0,
                                  1); // The 0 and 1 values were taken from example. It was also a subject for fix.

    std::vector<float> embeddings(n_sequences * embedding_rank_, 0);

    int processed_sequences = 0;
    int sequences_in_current_batch = 0;

    auto get_next_output_span = [&embeddings, &processed_sequences, this]()
    { return std::span<float>(embeddings.begin() + processed_sequences * embedding_rank_, embedding_rank_); };

    for(int k = 0; k < n_sequences; k++)
    {
        const auto& tokens = tokenized_sequences[k];
        const uint64_t n_toks = tokens.size();

        if(batch.n_tokens + n_toks > max_batch_)
        {
            batch_decode_(batch, get_next_output_span());
            llama_batch_clear(batch);
            processed_sequences += sequences_in_current_batch;
            sequences_in_current_batch = 0;
        }

        for(size_t i = 0; i < tokens.size(); i++)
        {
            llama_batch_add(batch, tokens[i], i, {sequences_in_current_batch},
                            i == tokens.size() - 1); // the last condition was taken directly from sample
        }
        sequences_in_current_batch++;
    }

    batch_decode_(batch, get_next_output_span());

    return embeddings;
}

void LLamaEmbeddingCalculator::free_llama_pointers()
{
    if(ctx)
    {
        llama_free(ctx);
        ctx = nullptr;
    }
    if(model)
    {
        llama_free_model(model);
        model = nullptr;
    }
}

std::unique_ptr<IEmbeddingCalculator> embedding_calculator_factory(const EmbeddingCalculatorParams& params)
{
    return std::make_unique<LLamaEmbeddingCalculator>(params);
}

} // namespace ds