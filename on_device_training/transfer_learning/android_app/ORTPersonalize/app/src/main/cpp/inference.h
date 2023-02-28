//
// Created by bmeswani on 2/17/2023.
//

#ifndef ORT_PERSONALIZE_INFERENCE_H
#define ORT_PERSONALIZE_INFERENCE_H

#include <cstdint>
#include <string>

#include "session_cache.h"

namespace inference {

    std::pair<std::string, float> classify(
            SessionCache* session_cache, float *image_data, int64_t batch_size, int64_t image_channels,
            int64_t image_rows, int64_t image_cols, const std::vector<std::string>& classes);

} // namespace inference

#endif //ORT_PERSONALIZE_INFERENCE_H
