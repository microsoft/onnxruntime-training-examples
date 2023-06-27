//
// Created by bmeswani on 2/17/2023.
//

#ifndef ORT_PERSONALIZE_INFERENCE_H
#define ORT_PERSONALIZE_INFERENCE_H

#include <cstdint>
#include <string>

#include "session_cache.h"

namespace inference {

    // runs the inference graph and returns a pair:
    //   - prediction string: one of the classes provided
    //   - probability: associated with the prediction of that class
    std::pair<std::string, float> classify(
            SessionCache* session_cache, float *image_data, int64_t batch_size, int64_t image_channels,
            int64_t image_rows, int64_t image_cols, const std::vector<std::string>& classes);

} // namespace inference

#endif //ORT_PERSONALIZE_INFERENCE_H