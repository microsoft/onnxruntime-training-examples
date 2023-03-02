//
// Created by bmeswani on 2/20/2023.
//

#include "train.h"


namespace training {

    float train_step(SessionCache* session_cache, float *batches, int32_t *labels,
                     int64_t batch_size, int64_t image_channels, int64_t image_rows,
                     int64_t image_cols) {
        const std::vector<int64_t> input_shape({batch_size, image_channels, image_rows, image_cols});
        const std::vector<int64_t> labels_shape({batch_size});

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> user_inputs; // {inputs, labels}
        // inputs batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, batches,
                                                          batch_size * image_channels * image_rows * image_cols * sizeof(float),
                                                          input_shape.data(), input_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // labels batched
        user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels,
                                                          batch_size * sizeof(int32_t),
                                                          labels_shape.data(), labels_shape.size(),
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));

        // run the train step and execute the forward + loss + backward.
        float loss = *(session_cache->training_session.TrainStep(user_inputs).front().GetTensorMutableData<float>());

        // update the model parameters by taking a step in the direction of the gradients computed above.
        session_cache->training_session.OptimizerStep();

        // reset the gradients now that the parameters have been updated.
        // new set of gradients can then be computed for the next round of inputs.
        session_cache->training_session.LazyResetGrad();

        return loss;
    }

} // namespace training