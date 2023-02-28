//
// Created by bmeswani on 2/20/2023.
//

#include "train.h"


namespace training {

    float train_step(SessionCache* session_cache, float *batches, int32_t *labels,
                     int64_t batch_size, int64_t image_channels, int64_t image_rows,
                     int64_t image_cols) {
        std::vector<int64_t> input_shape({batch_size, image_channels, image_rows, image_cols});
        std::vector<int64_t> labels_shape({batch_size});

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> user_inputs;
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, batches,
                                                          batch_size * image_channels * image_rows * image_cols * sizeof(float),
                                                          input_shape.data(), input_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels,
                                                          batch_size * sizeof(int32_t),
                                                          labels_shape.data(), labels_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
        float loss = *(session_cache->training_session.TrainStep(user_inputs).front().GetTensorMutableData<float>());

        session_cache->training_session.OptimizerStep();
        session_cache->training_session.LazyResetGrad();

        return loss;
    }

} // namespace training