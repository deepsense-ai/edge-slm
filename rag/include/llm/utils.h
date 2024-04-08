#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <type_traits>

#include <spdlog/spdlog.h>

namespace ds
{

template <typename Func, typename... Args>
concept is_result_void = std::is_void_v<std::invoke_result_t<Func, Args...>>;

template <typename Func>
auto with_time_report(std::string_view step_name, Func function)
{
    const auto start_ts = std::chrono::high_resolution_clock::now();
    const auto result = function();
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    spdlog::debug("Timing step: {}, took: {:.3f} ms", step_name, duration);

    return result;
}

template <is_result_void Func>
auto with_time_report(std::string_view step_name, Func function)
{
    const auto start_ts = std::chrono::high_resolution_clock::now();
    function();
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    spdlog::debug("Timing step: {}, took: {:.3f} ms", step_name, duration);
}

template <typename Func, typename... Args>
auto with_time_report(std::string_view step_name, Func function, Args... args)
{

    const auto start_ts = std::chrono::high_resolution_clock::now();
    const auto result = function(std::forward<Args>(args)...);
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    spdlog::debug("Timing step: {}, took: {:.3f} ms", step_name, duration);

    return result;
}

template <is_result_void Func, typename... Args>
void with_time_report(std::string_view step_name, Func function, Args... args)
{

    const auto start_ts = std::chrono::high_resolution_clock::now();
    function(std::forward<Args>(args)...);
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;
    const auto duration = (end_ts - start_ts) / 1.0ms;

    spdlog::debug("Timing step: {}, took: {:.3f} ms", step_name, duration);
}

template <typename Func, typename... Args>
std::tuple<std::invoke_result_t<Func, Args...>, float> with_time_measure(Func function, Args... args)
{
    const auto start_ts = std::chrono::high_resolution_clock::now();
    const auto result = function(std::forward<Args>(args)...);
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;

    const float total_seconds = (end_ts - start_ts) / 1us * 1e-6f;

    return std::make_tuple(std::move(result), total_seconds);
}

template <is_result_void Func>
float with_time_measure(Func function)
{
    const auto start_ts = std::chrono::high_resolution_clock::now();
    function();
    const auto end_ts = std::chrono::high_resolution_clock::now();

    using namespace std::chrono_literals;

    return (end_ts - start_ts) / 1us * 1e-6f;
}

std::filesystem::path get_configuration_directory();
} // namespace ds
