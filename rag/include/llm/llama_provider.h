#pragma once
#include "llm/llm_provider.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <istream>
#include <optional>

namespace ds
{

struct LlamaParameters
{
    float temp = 0.8f;
    uint32_t max_tokens = std::numeric_limits<uint32_t>::max();
    uint32_t context_size = 0;
    uint32_t threads = 1;
    std::optional<uint32_t> seed;
};

class LlamaProvider : public ILlmProvider
{
  public:
    static std::unique_ptr<LlamaProvider> from_config(const std::filesystem::path& model_path,
                                                      const std::filesystem::path& config_path);
    static std::unique_ptr<LlamaProvider> from_config(const std::filesystem::path& model_path,
                                                      std::istream& config_iss);

    LlamaProvider(const std::filesystem::path& model_path, const LlamaParameters& params);
    LlmOutput generate(const LlmInput& input) override;

    std::unique_ptr<ILlmAsyncGeneration> generateAsync(const LlmInput& input,
                                                       const std::function<LlmCallback>& callback) override;

    void clear_context() override;
    ~LlamaProvider() override;

  private:
    class LlamaProviderPimpl;
    std::unique_ptr<LlamaProviderPimpl> pimpl;
};

} // namespace ds
