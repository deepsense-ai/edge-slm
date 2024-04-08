#pragma once

#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <magic_enum/magic_enum.hpp>
#include <string>

#include <iostream>
#include <optional>

namespace ds
{

enum class ProgramMode {
    CHAT,
    RETRIEVAL
};

struct Options
{
    std::string embedding_model_path;
    std::string database_input;
    std::string database_output;
    std::string queries_input;
    std::string result_output;

    std::string prompt_template_path;
    std::string model_path;
    std::string model_config_path;

    int32_t embedding_threads;
    int32_t embedding_batch_size;
    uint32_t top_k;

    ProgramMode mode;
};

class OptionParser
{
  public:
    static Options parse_options(int argc, char* argv[])
    {
        Options opts;

        namespace po = boost::program_options;
        po::options_description description("Utility application for creating the embeddings database dump");

        std::string mode;

        description.add_options()("help,h", "produce help message");
        description.add_options()("embedding_model,m",
            po::value<std::string>(&opts.embedding_model_path)->default_value("./gte-base-f32.gguf"),
            "Model used to generate the embeddings.");
        description.add_options()("database_output,o", po::value<std::string>(&opts.database_output),
                                  "Path to an output file with calculated embeddings.");
        description.add_options()("queries_input,qi", po::value<std::string>(&opts.queries_input),
            "Input file containing the queries to run against. If provided interactive mode will be disabled.");
        description.add_options()("database_input,di", po::value<std::string>(&opts.database_input),
                                  "Input file containing the previously dumped database.");
        description.add_options()("embedding_threads,t", po::value<int32_t>(&opts.embedding_threads)->default_value(1),
                                  "Number of threads to run the embeddings model.");
        description.add_options()("result_output,ro", po::value<std::string>(&opts.result_output),
                                  "Output file to log the retrieved chunks with the queries");
        description.add_options()("embedding_batch_size,b", po::value<int32_t>(&opts.embedding_batch_size)->default_value(512),
                                  "Maximum batch size of running the embeddings model. Must be set to a value greater than n_ctx of the model.");
        description.add_options()("top_k,tk", po::value<uint32_t>(&opts.top_k)->default_value(3),
                                  "Maximum value of the returned document chunks per query.");
        description.add_options()("mode", po::value<std::string>(&mode)->default_value("CHAT"),
                                  "Set the application mode. Allowed values: {CHAT, RETRIEVAL}.");
        description.add_options()("prompt_template_path", po::value<std::string>(&opts.prompt_template_path),
                                  "A path to mustache prompt template");
        description.add_options()("model_path", po::value<std::string>(&opts.model_path),
                                  "A path to a LLM model in GGUF format.");
        description.add_options()("model_config_path", po::value<std::string>(&opts.model_config_path),
                                  "A path to a json LLM model configuration.");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, description), vm);
        po::notify(vm);

        if(vm.count("help"))
        {
            std::cout << description << std::endl;
            exit(1);
        }

        auto mode_enum = magic_enum::enum_cast<ProgramMode>(mode);
        if (!mode_enum) {
            throw po::validation_error(po::validation_error::invalid_option_value, "mode");
        }
        opts.mode = *mode_enum;

        return opts;
    }
};
} // namespace ds