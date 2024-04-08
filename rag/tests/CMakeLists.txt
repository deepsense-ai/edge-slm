find_package(GTest REQUIRED)

add_executable(rag_test
    src/rag/vector_database_test.cpp
    src/rag/embedding_calculator_test.cpp
    src/rag/document_retrieval_test.cpp
    src/rag/llm_prompt_composer.cpp
)

target_include_directories(rag_test PRIVATE ../include)
target_link_libraries(rag_test rag gtest::gtest)

add_test(NAME rag_test COMMAND rag_test)

add_executable(llm_test
    src/llm/llm_test.cpp)

target_include_directories(llm_test PRIVATE ../include)
target_link_libraries(llm_test llm gtest::gtest)

add_test(NAME llm_test COMMAND llm_test)

install(TARGETS rag_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS llm_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
