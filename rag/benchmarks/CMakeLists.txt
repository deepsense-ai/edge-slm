find_package(benchmark REQUIRED)

add_executable(vector_db_benchmark
    src/vector_search.cpp
)


target_include_directories(vector_db_benchmark PRIVATE ../include)
target_link_libraries(vector_db_benchmark rag benchmark::benchmark_main)

add_executable(embeddings_benchmark
    src/embedding_calculator.cpp
)

target_include_directories(embeddings_benchmark PRIVATE ../include)
target_link_libraries(embeddings_benchmark rag benchmark::benchmark_main)

add_executable(llm_benchmark
    src/llm_benchmark.cpp
)
target_include_directories(llm_benchmark PRIVATE ../include)
target_link_libraries(llm_benchmark llm benchmark::benchmark_main)

install(TARGETS embeddings_benchmark vector_db_benchmark llm_benchmark
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
