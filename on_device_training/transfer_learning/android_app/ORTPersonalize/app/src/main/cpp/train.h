//
// Created by bmeswani on 2/20/2023.
//

#ifndef ORT_PERSONALIZE_TRAIN_H
#define ORT_PERSONALIZE_TRAIN_H

#include "onnxruntime_training_cxx_api.h"
#include "session_cache.h"

namespace training {

    float train_step(SessionCache* session_cache, float *batches, int32_t *labels,
                     int64_t batch_size, int64_t image_channels, int64_t image_rows,
                     int64_t image_cols);

} // namespace training

#endif //ORT_PERSONALIZE_TRAIN_H
