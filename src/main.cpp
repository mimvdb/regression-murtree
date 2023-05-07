#include "regression.cpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: regression <input_file>");
        return -1;
    }

    const char* input_path = argv[1];
    std::vector<Instance> instances;
    read_instances(input_path, instances);
    return 0;
}