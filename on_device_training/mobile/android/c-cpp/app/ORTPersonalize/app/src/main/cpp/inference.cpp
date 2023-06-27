//
// Created by bmeswani on 2/17/2023.
//

#include "inference.h"
#include <cassert>

#include "onnxruntime_cxx_api.h"

#include <android/log.h>

namespace {

    std::vector<float> Softmax(float *logits, size_t num_logits) {
        std::vector<float> probabilities(num_logits, 0);
        float sum = 0;
        for (size_t i = 0; i < num_logits; ++i) {
            probabilities[i] = exp(logits[i]);
            sum += probabilities[i];
        }

        if (sum != 0.0f) {
            for (size_t i = 0; i < num_logits; ++i) {
                probabilities[i] /= sum;
            }
        }

        return probabilities;
    }

} //namespace

namespace inference {

    std::pair<std::string, float> classify(SessionCache* session_cache, float *image_data,
                                           int64_t batch_size, int64_t image_channels,
                                           int64_t image_rows, int64_t image_cols,
                                           const std::vector<std::string>& classes) {
        std::vector<const char *> input_names = {"input"};
        size_t input_count = 1;

        std::vector<const char *> output_names = {"output"};
        size_t output_count = 1;

        std::vector<int64_t> input_shape({batch_size, image_channels, image_rows, image_cols});

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_values; // {input images}
        input_values.emplace_back(Ort::Value::CreateTensor(memory_info, image_data,
                                                           batch_size * image_channels * image_rows * image_cols * sizeof(float),
                                                           input_shape.data(), input_shape.size(),
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));


        std::vector<Ort::Value> output_values;
        output_values.emplace_back(nullptr);

        // get the logits
        session_cache->inference_session->Run(Ort::RunOptions(), input_names.data(), input_values.data(),
                                              input_count, output_names.data(), output_values.data(), output_count);

        float *output = output_values.front().GetTensorMutableData<float>();

        // run softmax and get the probabilities of each class
        std::vector<float> probabilities = Softmax(output, classes.size());
        size_t best_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

        return {classes[best_index], probabilities[best_index]};
    }

} // namespace inference
