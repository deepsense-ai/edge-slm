#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <utility>

namespace ds
{

struct LlmInput
{
    std::string prompt;
};

struct LlmTimings
{
    float tokens_per_second;
    float generation_time_seconds;
    float tokenization_time_seconds;
    float prompt_decoding_time_seconds;
};

struct LlmOutput
{
    std::string answer;
    LlmTimings timings;
};

enum class AsyncStatus
{
    GENERATING,
    FINISHED,
    STOPPED
};

struct LlmAsyncOutput : LlmOutput
{
    AsyncStatus status;
};

class ILlmAsyncGeneration
{
  public:
    virtual std::future<LlmOutput> start() = 0;
    virtual void stop() = 0;
    virtual ~ILlmAsyncGeneration() = default;
};

using LlmCallback = void(const LlmAsyncOutput&);

class ILlmProvider
{
  public:
    virtual LlmOutput generate(const LlmInput& input) = 0;
    virtual std::unique_ptr<ILlmAsyncGeneration> generateAsync(const LlmInput& input,
                                                               const std::function<LlmCallback>& callback) = 0;
    virtual void clear_context() = 0;
    virtual ~ILlmProvider() = default;
};

} // namespace ds
