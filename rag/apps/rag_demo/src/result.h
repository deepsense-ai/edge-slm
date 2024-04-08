#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ds {

    struct RetrievalResult{
        std::string query;
        std::vector<std::string> contexts;
    };

    void to_json(nlohmann::json& j, const RetrievalResult& r)
    {
        j = nlohmann::json{ {"query", r.query}, {"contexts", r.contexts} };
    }
}