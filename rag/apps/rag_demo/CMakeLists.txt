find_package(tabulate REQUIRED)
find_package(magic_enum REQUIRED)

add_executable(
    rag_demo
    src/main.cpp
)

target_link_libraries(rag_demo PRIVATE
    rag
    llm
    spdlog::spdlog
    boost::boost
    fmt::fmt
    tabulate::tabulate
    nlohmann_json::nlohmann_json
    magic_enum::magic_enum
)

install(TARGETS rag_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
