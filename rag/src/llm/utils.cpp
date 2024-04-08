#include "llm/utils.h"
#include <filesystem>
#include <cstdlib>

namespace ds {

std::filesystem::path get_configuration_directory() {
    auto env_val = std::getenv("DS_CONFIG_DIR");
    if (env_val == nullptr)
        return "./config";
    else
        return env_val;
}

}