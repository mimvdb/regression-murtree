#include <vector>
#include <stdint.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

struct Instance {
    double target;
    // Allow a maximum of 64 features, replace by bitset to allow for more. (e.g. with a struct hack array)
    uint64_t features;
};

static bool is_set(uint64_t bitset, int nth) {
    assert(nth < 64);
    return (bitset & (UINT64_C(1) << nth)) > 0;
}

static void bit_set(uint64_t& bitset, int nth) {
    assert(nth < 64);
    bitset = bitset | (UINT64_C(1) << nth);
}

static void read_instances(const char* path, std::vector<Instance>& instances) {
    std::ifstream input_stream(path);
    std::string line;
    while (std::getline(input_stream, line)) {
        Instance new_instance = {};
        std::istringstream iss(line);
        iss >> new_instance.target;
        // First column is the target, followed by some number of binary features
        int feature;
        int nth_feature = 0;
        while (iss >> feature) {
            assert(feature == 0 || feature == 1);
            if (feature) bit_set(new_instance.features, nth_feature);
            nth_feature++;
        }
        assert(nth_feature < 64);
        instances.push_back(new_instance);
    }
}