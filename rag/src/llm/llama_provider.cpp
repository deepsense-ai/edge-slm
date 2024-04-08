#include "llm/llama_provider.h"
#include "llm/llm_provider.h"
#include "llm/utils.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <llamacpp/common.h>
#include <llamacpp/llama.h>
#include <llamacpp/sampling.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <span>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace ds
{

template <typename T>
using unique_ptr_with_deleter = std::unique_ptr<T, std::function<void(T*)>>;

class StopToken
{
  public:
    bool stop_requested() { return flag_.load(); }

  private:
    StopToken(std::atomic_bool& flag) : flag_{flag} {}

    std::atomic_bool& flag_;

    friend class StopSource;
};

class StopSource
{
  public:
    bool request_stop() noexcept
    {
        flag_ = true;
        return flag_.load();
    }

    StopToken get_token() noexcept { return StopToken{flag_}; }

  private:
    std::atomic_bool flag_ = false;
};

class LlamaModel
{
  public:
    explicit LlamaModel(std::filesystem::path model_path, const llama_model_params& model_params);

    llama_model* get_llama_model();

  private:
    unique_ptr_with_deleter<llama_model> model;
};

class LlamaSampling
{
  public:
    LlamaSampling(llama_sampling_params params);

    llama_sampling_context* get_sampling_context();

  private:
    unique_ptr_with_deleter<llama_sampling_context> context_;
};

struct LlamaToken
{
    llama_token id;
    std::string text;
    bool is_eos;
};

class LlamaContext
{
  public:
    LlamaContext(std::unique_ptr<LlamaModel> model, std::unique_ptr<LlamaSampling> sampling,
                 const llama_context_params& context_params);

    LlmOutput generate(const LlmInput& input, std::function<LlmCallback> callback = std::function<LlmCallback>(),
                       std::optional<StopToken> stop = std::nullopt,
                       uint32_t max_tokens = std::numeric_limits<uint32_t>::max());

    void clear_context();

  private:
    std::vector<llama_token> tokenize(const std::string& text, bool special);
    void decode(const std::span<llama_token> embeddings);
    LlamaToken get_next_token();
    void warmup_model();
    void prepare_context(const std::span<llama_token> embeddings);
    void swap_context();

    std::unique_ptr<LlamaModel> model_;
    std::unique_ptr<LlamaSampling> sampling_;
    unique_ptr_with_deleter<llama_context> context_;
    llama_context_params context_params_;

    llama_pos n_past = 0;
};

class LlamaAsyncGeneration : public ILlmAsyncGeneration
{
  public:
    LlamaAsyncGeneration(std::shared_ptr<LlamaContext> context, const LlmInput& input,
                         std::function<LlmCallback> callback, uint32_t max_tokens);

    std::future<LlmOutput> start() override;
    void stop() override;

  private:
    std::shared_ptr<LlamaContext> context_;
    LlmInput input_;
    std::function<LlmCallback> callback_;

    StopSource stop_;
    uint32_t max_tokens_;
};

class LlamaProvider::LlamaProviderPimpl
{
  public:
    LlamaProviderPimpl(const std::filesystem::path& model_path, const LlamaParameters& params);
    LlmOutput generate(const LlmInput& input);
    std::unique_ptr<ILlmAsyncGeneration> generateAsync(const LlmInput& input,
                                                       const std::function<LlmCallback>& callback);

    void clear_context();
    ~LlamaProviderPimpl();

  private:
    llama_model_params get_model_params() const;
    llama_context_params get_context_params() const;
    llama_sampling_params get_sampling_params() const;

    std::shared_ptr<LlamaContext> context_;
    LlamaParameters params_;
};

LlamaContext::LlamaContext(std::unique_ptr<LlamaModel> model, std::unique_ptr<LlamaSampling> sampling,
                           const llama_context_params& context_params)
    : model_{std::move(model)}, sampling_{std::move(sampling)}, context_params_{context_params}
{
    context_ = unique_ptr_with_deleter<llama_context>(
        llama_new_context_with_model(model_->get_llama_model(), context_params), &llama_free);

    warmup_model();
}

LlmOutput LlamaContext::generate(const LlmInput& input, std::function<LlmCallback> callback,
                                 std::optional<StopToken> stop, uint32_t max_tokens)
{
    auto [embeddings, tokenization_time_seconds] =
        with_time_measure(std::bind_front(&LlamaContext::tokenize, this), input.prompt, true);

    std::span<llama_token> embeddings_span(embeddings.begin(), embeddings.end());
    auto prompt_decoding_time_seconds = with_time_measure([this, &embeddings_span]() {decode(embeddings_span);});

    uint32_t total_tokens = 0;
    std::string full_answer;

    auto generation_fn = [&, this]()
    {
        bool is_running = true;
        while(is_running)
        {
            const auto token = get_next_token();

            full_answer += token.text;

            const bool is_stop_requested = stop.has_value() && stop->stop_requested();
            const bool is_max_tokens = ++total_tokens >= max_tokens;
            is_running = !token.is_eos && !is_stop_requested && !is_max_tokens;

            if(callback)
            {
                LlmAsyncOutput partial_result;
                partial_result.status = token.is_eos || is_max_tokens
                                            ? AsyncStatus::FINISHED
                                            : (is_stop_requested ? AsyncStatus::STOPPED : AsyncStatus::GENERATING);
                partial_result.answer = token.text;

                callback(partial_result);
            }

            embeddings = {token.id};
            embeddings_span = std::span<llama_token>(embeddings.begin(), embeddings.end());
            decode(embeddings_span);
        }
    };

    using namespace std::chrono_literals;
    const float total_time = with_time_measure(generation_fn);
    const LlmTimings timings = {.tokens_per_second = total_tokens / total_time,
                                .generation_time_seconds = total_time,
                                .tokenization_time_seconds = tokenization_time_seconds,
                                .prompt_decoding_time_seconds = prompt_decoding_time_seconds};
    const LlmOutput result = {.answer = std::move(full_answer), .timings = timings};

    return result;
}

void LlamaContext::clear_context()
{
    llama_kv_cache_clear(context_.get());
    n_past = 0;
}

std::vector<llama_token> LlamaContext::tokenize(const std::string& text, bool special)
{
    const int add_bos_int = llama_add_bos_token(model_->get_llama_model());
    const bool add_bos =
        add_bos_int != -1 ? bool(add_bos_int) : (llama_vocab_type(model_->get_llama_model()) == LLAMA_VOCAB_TYPE_SPM);

    int num_of_tokens = text.length() + add_bos;
    std::vector<llama_token> result(num_of_tokens);

    num_of_tokens = llama_tokenize(model_->get_llama_model(), text.data(), text.length(), result.data(), result.size(),
                                   add_bos, special);

    if(num_of_tokens < 0)
    {
        num_of_tokens = -num_of_tokens;
        int check = llama_tokenize(model_->get_llama_model(), text.data(), text.length(), result.data(), result.size(),
                                   add_bos, special);

        GGML_ASSERT(check == num_of_tokens);
    }

    result.resize(num_of_tokens);

    return result;
}

void LlamaContext::decode(const std::span<llama_token> embeddings)
{
    prepare_context(embeddings);

    auto n_eval = std::min(embeddings.size(), static_cast<std::size_t>(context_params_.n_batch));
    llama_decode(context_.get(), llama_batch_get_one(embeddings.data(), n_eval, n_past, 0));

    n_past += n_eval;
}

LlamaToken LlamaContext::get_next_token()
{
    const llama_token id = llama_sampling_sample(sampling_->get_sampling_context(), context_.get(), nullptr);
    llama_sampling_accept(sampling_->get_sampling_context(), context_.get(), id, true);

    const auto token_str = llama_token_to_piece(context_.get(), id);
    LlamaToken result = {.id = id, .text = token_str, .is_eos = llama_token_eos(model_->get_llama_model()) == id};

    return result;
}

void LlamaContext::warmup_model()
{
    auto native_model = model_->get_llama_model();

    std::vector<llama_token> tmp = {llama_token_bos(native_model), llama_token_eos(native_model)};

    llama_decode(
        context_.get(),
        llama_batch_get_one(tmp.data(), std::min(tmp.size(), static_cast<std::size_t>(context_params_.n_batch)), 0, 0));

    clear_context();
    llama_reset_timings(context_.get());
}

void LlamaContext::prepare_context(const std::span<llama_token> embeddings)
{
    auto required_size = n_past + embeddings.size();
    const auto n_context_tokens = llama_n_ctx(context_.get());

    if(required_size > n_context_tokens)
    {
        swap_context();
    }
}

void LlamaContext::swap_context()
{
    const auto n_keep = 0;

    const int n_left = n_past - n_keep - 1;
    const int n_discard = n_left / 2;

    llama_kv_cache_seq_rm(context_.get(), 0, n_keep + 1, n_keep + n_discard + 1);
    llama_kv_cache_seq_add(context_.get(), 0, n_keep + 1 + n_discard, n_past, -n_discard);

    n_past -= n_discard;
}

LlamaModel::LlamaModel(std::filesystem::path model_path, const llama_model_params& model_params)
{
    model = unique_ptr_with_deleter<llama_model>(llama_load_model_from_file(model_path.c_str(), model_params),
                                                 &llama_free_model);
}

llama_model* LlamaModel::get_llama_model()
{
    return model.get();
}

LlamaSampling::LlamaSampling(llama_sampling_params params)
{
    context_ = unique_ptr_with_deleter<llama_sampling_context>(llama_sampling_init(params), &llama_sampling_free);
}

llama_sampling_context* LlamaSampling::get_sampling_context()
{
    return context_.get();
}

LlamaProvider::LlamaProviderPimpl::LlamaProviderPimpl(const std::filesystem::path& model_path,
                                                      const LlamaParameters& params)
    : params_{params}
{
    llama_backend_init();

    auto model = std::make_unique<LlamaModel>(model_path, get_model_params());
    auto sampling = std::make_unique<LlamaSampling>(get_sampling_params());

    context_ = std::make_shared<LlamaContext>(std::move(model), std::move(sampling), get_context_params());
}

LlmOutput LlamaProvider::LlamaProviderPimpl::generate(const LlmInput& input)
{
    return context_->generate(input, std::function<LlmCallback>(), std::nullopt, params_.max_tokens);
}

std::unique_ptr<ILlmAsyncGeneration>
LlamaProvider::LlamaProviderPimpl::generateAsync(const LlmInput& input, const std::function<LlmCallback>& callback)
{
    return std::make_unique<LlamaAsyncGeneration>(context_, input, callback, params_.max_tokens);
}

void LlamaProvider::LlamaProviderPimpl::clear_context()
{
    return context_->clear_context();
}

LlamaProvider::LlamaProviderPimpl::~LlamaProviderPimpl()
{
    context_.reset();

    llama_backend_free();
}

llama_model_params LlamaProvider::LlamaProviderPimpl::get_model_params() const
{
    auto params = llama_model_default_params();

    params.use_mlock = true;
    params.use_mmap = false;

    return params;
}

llama_context_params LlamaProvider::LlamaProviderPimpl::get_context_params() const
{
    auto params = llama_context_default_params();

    params.seed = params_.seed ? params_.seed.value() : time(NULL);
    params.n_ctx = params_.context_size;
    params.n_threads = params_.threads;

    return params;
}

llama_sampling_params LlamaProvider::LlamaProviderPimpl::get_sampling_params() const
{
    llama_sampling_params params{};

    params.temp = params_.temp;

    return params;
}

std::unique_ptr<ILlmAsyncGeneration> LlamaProvider::generateAsync(const LlmInput& input,
                                                                  const std::function<LlmCallback>& callback)
{
    return pimpl->generateAsync(input, callback);
}

void LlamaProvider::clear_context()
{
    pimpl->clear_context();
}

LlamaProvider::LlamaProvider(const std::filesystem::path& model_path, const LlamaParameters& params)
    : pimpl{std::make_unique<LlamaProviderPimpl>(model_path, params)}
{
}

LlmOutput LlamaProvider::generate(const LlmInput& input)
{
    return pimpl->generate(input);
}

LlamaProvider::~LlamaProvider() = default;

LlamaAsyncGeneration::LlamaAsyncGeneration(std::shared_ptr<LlamaContext> context, const LlmInput& input,
                                           std::function<LlmCallback> callback, uint32_t max_tokens)
    : context_{std::move(context)}, input_{input}, callback_{std::move(callback)}, max_tokens_{max_tokens}
{
}

std::future<LlmOutput> LlamaAsyncGeneration::start()
{
    return std::async(std::launch::async, [this, stop_token = stop_.get_token()]()
                      { return context_->generate(input_, callback_, stop_token, max_tokens_); });
}

void LlamaAsyncGeneration::stop()
{
    stop_.request_stop();
}

std::unique_ptr<LlamaProvider> LlamaProvider::from_config(const std::filesystem::path& model_path,
                                                          const std::filesystem::path& config_path)
{
    std::ifstream config_iss{config_path};
    return from_config(model_path, config_iss);
}

template <typename T>
std::optional<T> get_optional(const std::string& key, const nlohmann::json& j,
                              const std::optional<T>& default_value = std::nullopt)
{
    if(auto findIt = j.find(key); findIt != std::end(j))
        return findIt->is_null() ? std::nullopt : std::make_optional(findIt->get<T>());

    return default_value;
}

void from_json(const nlohmann::json& j, LlamaParameters& params)
{
    params = LlamaParameters{};
    params.temp = j.value("temp", params.temp);

    params.max_tokens = j.value("max_tokens", params.max_tokens);
    params.context_size = j.value("context_size", params.context_size);
    params.threads = j.value("threads", params.threads);
    params.seed = get_optional("seed", j, params.seed);
}

std::unique_ptr<LlamaProvider> LlamaProvider::from_config(const std::filesystem::path& model_path,
                                                          std::istream& config_iss)
{
    const auto json_config = nlohmann::json::parse(config_iss);

    LlamaParameters params;
    json_config.get_to(params);

    return std::make_unique<LlamaProvider>(model_path, params);
}

} // namespace ds
