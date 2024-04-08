#pragma once

#ifndef TEST_ASSETS_DIR
#error No definition of TEST_ASSETS_DIR found
#endif

#include <filesystem>

namespace ds
{
const std::filesystem::path ASSETS_ROOT = std::filesystem::path(TEST_ASSETS_DIR);

} // namespace ds

