#include "model_runner.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<float> read_float_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>()};
    std::vector<float> data(bytes.size() / sizeof(float));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

void write_float_file(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
    // usage: model_example <model.axmodel> <input_0.bin> [input_1.bin ...] <output_dir>
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: %s <model.axmodel> <input_0.bin> [input_1.bin ...] <output_dir>\n",
            argv[0]);
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        const std::string output_dir = argv[argc - 1];
        std::vector<std::vector<float>> inputs;
        for (int i = 2; i < argc - 1; ++i) {
            inputs.push_back(read_float_file(argv[i]));
        }
        ModelRunner runner(model_path, "model");
        std::vector<std::vector<float>> outputs = runner.Run(inputs);
        for (size_t i = 0; i < outputs.size(); ++i) {
            write_float_file(
                output_dir + "/output_" + std::to_string(i) + ".bin", outputs[i]);
        }
        std::printf("inputs=%zu outputs=%zu\n", runner.NumInputs(), runner.NumOutputs());
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
