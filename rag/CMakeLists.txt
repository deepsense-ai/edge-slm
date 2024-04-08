cmake_minimum_required(VERSION 3.15)
project(rag CXX)

set(CMAKE_CXX_STANDARD 20)

include(CTest)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)
find_package(llamacpp REQUIRED)
find_package(Boost REQUIRED)
find_package(kainjow_mustache REQUIRED)

add_library(llm SHARED
    src/llm/llama_provider.cpp
    src/llm/utils.cpp
)

target_link_libraries(llm PRIVATE
    spdlog::spdlog
    llamacpp::llamacpp
    nlohmann_json::nlohmann_json)

target_include_directories(llm PUBLIC include)

add_library(rag SHARED
    src/rag/faiss_vector_database.cpp
    src/rag/embedding_calculator.cpp
    src/rag/llama_embedding_calculator.cpp
    src/rag/document_retrieval.cpp
    src/rag/llm_prompt_composer.cpp
    src/rag/pipeline.cpp
)

target_link_libraries(rag PRIVATE
    spdlog::spdlog
    fmt::fmt
    boost::boost
    faiss::faiss
    OpenMP::OpenMP_CXX
    llamacpp::llamacpp
    nlohmann_json::nlohmann_json
    kainjow_mustache::kainjow_mustache
)

target_link_libraries(rag PUBLIC
    llm
)

target_include_directories(rag PUBLIC include)

if(ANDROID)
    add_compile_definitions(TEST_ASSETS_DIR="./assets")
else()
    add_compile_definitions(TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
endif(ANDROID)

add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(apps)

include(GNUInstallDirs)

install(TARGETS rag
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS llm
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/ DESTINATION ${CMAKE_INSTALL_PREFIX}/assets)
