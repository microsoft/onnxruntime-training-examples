#include <jni.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <sstream>
#include <android/log.h>

#include "onnxruntime_c_api.h"
#include "onnxruntime_training_cxx_api.h"

#define LOG_TAG "ondevicetraining"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace
{
    std::string JString2String(JNIEnv *env, jstring jStr)
    {
        if (!jStr)
            return "";

        const jclass stringClass = env->GetObjectClass(jStr);
        const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
        const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

        size_t length = (size_t)env->GetArrayLength(stringJbytes);
        jbyte *pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

        std::string ret = std::string((char *)pBytes, length);
        env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

        env->DeleteLocalRef(stringJbytes);
        env->DeleteLocalRef(stringClass);
        return ret;
    }

    std::vector<float> Softmax(float *logits)
    {
        std::vector<float> probabilities(10, 0);
        float sum = 0;
        for (size_t i = 0; i < 10U; ++i)
        {
            probabilities[i] = exp(logits[i]);
            sum += probabilities[i];
        }

        if (sum != 0.0f)
        {
            for (size_t i = 0; i < 10U; ++i)
            {
                probabilities[i] /= sum;
            }
        }

        return probabilities;
    }
}

struct DataLoader
{
    int step = 0;
    int group = 0;
    int max_group_size = 250;
    int max_step_size;
    int batched_input_size = 4 * 3 * 32 * 32;
    int batched_labels_size = 4;
    int input_step_size = max_group_size * batched_input_size;
    int labels_step_size = max_group_size * batched_labels_size;
    std::vector<float> input_data;
    std::vector<int32_t> labels_data;
    std::string data_path;

    DataLoader(int max_num_of_steps_available)
    {
        max_step_size = max_num_of_steps_available;
    }

    void SetDataSetPath(std::string path)
    {
        data_path = path;
    }

    int GetCurrentStep()
    {
        return step;
    }

    int GetMaxGroupSize()
    {
        return max_group_size;
    }

    template <class T>
    bool LoadData(std::string filename, std::vector<T> &input_buf, size_t expected_len)
    {
        std::ifstream data_file(filename, std::ios::binary);
        if (!data_file)
        {
            LOGE("Data file does not exist: %s.", filename.c_str());
            return false;
        }

        // get length of file:
        data_file.seekg(0, data_file.end);
        int length = data_file.tellg();
        data_file.seekg(0, data_file.beg);
        if (length != static_cast<int>(expected_len))
        {
            LOGE("Actual length of the data file is: %d. Expected length is %zu", length, expected_len);
            return false;
        }

        T val;
        while (data_file.read(reinterpret_cast<char *>(&val), sizeof(T)))
        {
            input_buf.push_back(val);
        }
        return true;
    }

    bool LoadDataSet(int step)
    {
        // Loads 1 step for train\test data set.
        std::string input_file_name = data_path + "/input_" + std::to_string(step) + ".bin";
        std::string labels_file_name = data_path + "/labels_" + std::to_string(step) + ".bin";

        input_data.clear();
        labels_data.clear();
        input_data.reserve(input_step_size);
        labels_data.reserve(labels_step_size);
        auto data_loaded = LoadData<float>(input_file_name, input_data, input_step_size * sizeof(float));
        if (!data_loaded)
        {
            return false;
        }
        data_loaded = LoadData<int>(labels_file_name, labels_data, labels_step_size * sizeof(int));
        if (!data_loaded)
        {
            return false;
        }
        return true;
    }

    void GetData(std::vector<float> &input, std::vector<int32_t> &labels)
    {
        if (input_data.empty() && !LoadDataSet(step))
        {
            return;
        }

        // Load data for the current batch
        input.insert(input.begin(), (input_data.begin() + group * batched_input_size),
                     (input_data.begin() + (group + 1) * batched_input_size));

        labels.insert(labels.begin(), (labels_data.begin() + group * batched_labels_size),
                      (labels_data.begin() + (group + 1) * batched_labels_size));

        group++;

        // go to the next step
        if (input_data.end() == input_data.begin() + group * batched_input_size)
        {
            group = 0;
            input_data.clear();
            labels_data.clear();
            step++;
        }
    }
};

struct TrainingSessionCache
{
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    Ort::CheckpointState checkpoint_state;
    Ort::TrainingSession session;
    DataLoader data_loader;

    TrainingSessionCache(const std::string &checkpoint_path, const std::string &training_model_path,
                         const std::string &eval_model_path, const std::string &optimizer_model_path,
                         int dataloader_max_steps) : ort_env(ORT_LOGGING_LEVEL_VERBOSE, LOG_TAG), session_options(),
                                                     checkpoint_state(Ort::CheckpointState::LoadCheckpoint(checkpoint_path.c_str())),
                                                     session(session_options, checkpoint_state, training_model_path.c_str(), eval_model_path.c_str(), optimizer_model_path.c_str()),
                                                     data_loader(dataloader_max_steps) {}
};

// Native Functions

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ondevicetraining_MainActivity_getTrainingSessionCache(
    JNIEnv *env,
    jobject /* this */,
    jstring checkpoint_path,
    jstring train_model_path,
    jstring eval_model_path,
    jstring optimizer_model_path,
    jint dataloader_max_steps)
{
    std::unique_ptr<TrainingSessionCache> session_cache = std::make_unique<TrainingSessionCache>(
        JString2String(env, checkpoint_path),
        JString2String(env, train_model_path),
        JString2String(env, eval_model_path),
        JString2String(env, optimizer_model_path),
        dataloader_max_steps);

    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ondevicetraining_MainActivity_releaseTrainingResource(
    JNIEnv *env, jobject /* this */,
    jlong training_resource)
{
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(training_resource);
    delete session_cache;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ondevicetraining_MainActivity_infer(
    JNIEnv *env,
    jobject /* this */,
    jstring model_path,
    jfloatArray buffer,
    jint batch_size,
    jint channels,
    jint frame_cols,
    jint frame_rows,
    jlong training_resource)
{
    // This function returns the prediction from the given input as a string.

    // Export the eval model to an inference ready model at path provided.
    auto *session_cache = (TrainingSessionCache *)training_resource;
    const std::vector<std::string> graph_outputs({"output"});
    session_cache->session.ExportModelForInferencing(JString2String(env, model_path).c_str(), graph_outputs);

    // Create the inference session
    Ort::Env ort_env(ORT_LOGGING_LEVEL_VERBOSE, LOG_TAG);
    auto session_options = Ort::SessionOptions();
    auto inference_session = Ort::Session(ort_env, JString2String(env, model_path).c_str(), session_options);

    // Run a single inference
    float *input_data = env->GetFloatArrayElements(buffer, nullptr);
    int64_t input_shape[] = {batch_size, channels, frame_cols, frame_rows};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_values;
    input_values.emplace_back(Ort::Value::CreateTensor(memory_info, input_data, batch_size * channels * frame_cols * frame_rows * sizeof(float),
                                                       input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

    auto run_options = Ort::RunOptions();
    std::vector<const char *> input_names = {"input"};
    auto input_count = 1;
    std::vector<const char *> output_names = {"output"};
    auto output_count = 1;
    std::vector<Ort::Value> output_values;
    output_values.emplace_back(nullptr);
    inference_session.Run(run_options, input_names.data(), input_values.data(),
                          input_count, output_names.data(), output_values.data(), output_count);

    // Translate output to user recognizable class and return to the user
    float *output = output_values.front().GetTensorMutableData<float>();
    std::vector<float> probabilities = Softmax(output);
    size_t best_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

    std::vector<std::string> labels({"an airplane",
                                     "a car",
                                     "a bird",
                                     "a cat",
                                     "a deer",
                                     "a dog",
                                     "a frog",
                                     "a horse",
                                     "a ship",
                                     "a truck"});

    // Keep only 2 decimal places to display to user.
    std::ostringstream probability_stream;
    probability_stream.precision(2);
    probability_stream << std::fixed << probabilities[best_index] * 100;

    std::string label(labels[best_index] + " (" + probability_stream.str() + "%).");
    return env->NewStringUTF(label.c_str());
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ondevicetraining_MainActivity_train(
    JNIEnv *env,
    jobject /* this */,
    jlong training_resource,
    jstring cache_train_dataset_path)
{
    // This function returns a string "Success" or "Fail" to be displayed to the user
    // after training (one step) is completed.

    // Retrieve the training session and begin training for this round.
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(training_resource);
    auto &train_data_loader = session_cache->data_loader;
    train_data_loader.SetDataSetPath(JString2String(env, cache_train_dataset_path).c_str());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                             OrtMemTypeDefault);
    std::vector<float> input_data_vec;
    std::vector<int> labels_data_vec;
    input_data_vec.reserve(4 * 3 * 32 * 32);
    labels_data_vec.reserve(4);
    int64_t input_shape[] = {4, 3, 32, 32};
    int64_t labels_shape[] = {4};
    int max_iters = train_data_loader.GetMaxGroupSize();
    long duration = 0;
    int num_steps = 0;
    for (int iter = 0; iter < max_iters; iter++)
    {
        // Get Input Data
        input_data_vec.clear();
        labels_data_vec.clear();
        train_data_loader.GetData(input_data_vec, labels_data_vec);
        if (input_data_vec.size() == 0 || labels_data_vec.size() == 0)
        {
            if (iter == 0 && train_data_loader.GetCurrentStep() == 0)
            {
                return env->NewStringUTF("Training cannot start!! No data available");
            }
            else
            {
                return env->NewStringUTF("Training stopped no more data available");
            }
        }

        // Prepare the graph inputs
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor(memory_info, input_data_vec.data(),
                                                     4 * 3 * 32 * 32 * sizeof(float),
                                                     input_shape, 4,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels_data_vec.data(),
                                                     4 * sizeof(int), labels_shape, 1,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto fetches = session_cache->session.TrainStep(inputs); // train step = compute loss + compute gradients
        session_cache->session.OptimizerStep();                  // parameter update
        session_cache->session.ResetGrad();                      // reset gradients
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        num_steps++;
        float *loss = fetches[0].GetTensorMutableData<float>();
        LOGI("Training loss: %f for iteration %d", loss[0], iter);
    }

    std::string duration_string = "Average time for TrainStep is " + std::to_string(duration / num_steps) + " us.";
    return env->NewStringUTF(duration_string.c_str());
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ondevicetraining_MainActivity_eval(
    JNIEnv *env,
    jobject /* this */,
    jlong training_resource,
    jstring test_data_set_cache_dir)
{
    // This function returns the validation loss as a string.
    auto *session_cache = reinterpret_cast<TrainingSessionCache *>(training_resource);
    std::unique_ptr<DataLoader> eval_data_loader = std::make_unique<DataLoader>(10);
    eval_data_loader->SetDataSetPath(JString2String(env, test_data_set_cache_dir).c_str());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                             OrtMemTypeDefault);
    std::vector<float> input_data_vec;
    std::vector<int> labels_data_vec;
    input_data_vec.reserve(4 * 3 * 32 * 32);
    labels_data_vec.reserve(4);
    int64_t input_shape[] = {4, 3, 32, 32};
    int64_t labels_shape[] = {4};
    int max_iters = eval_data_loader->GetMaxGroupSize();
    float eval_loss = 0;
    long num_steps = 0;
    for (int iter = 0; iter < max_iters * eval_data_loader->max_step_size; iter++)
    {
        // Get Input Data
        input_data_vec.clear();
        labels_data_vec.clear();
        eval_data_loader->GetData(input_data_vec, labels_data_vec);

        if (input_data_vec.size() == 0 || labels_data_vec.size() == 0)
        {
            LOGW("Evaluation cannot be completed. No more data available.");
            break;
        }

        // Prepare the graph inputs
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor(memory_info, input_data_vec.data(),
                                                     4 * 3 * 32 * 32 * sizeof(float),
                                                     input_shape, 4,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        inputs.emplace_back(Ort::Value::CreateTensor(memory_info, labels_data_vec.data(),
                                                     4 * sizeof(int), labels_shape, 1,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));

        auto fetches = session_cache->session.EvalStep(inputs);
        num_steps++;
        eval_loss += *(fetches[0].GetTensorMutableData<float>());
        LOGW("Cumulative validation loss: %f for iter %d", eval_loss, iter);
    }

    float final_loss = num_steps ? eval_loss / num_steps : 0.0f;
    std::string validation_loss_str = "Validation loss " + std::to_string(eval_loss / num_steps);
    std::ostringstream loss_stream;
    loss_stream.precision(3);
    loss_stream << std::fixed << final_loss;
    return env->NewStringUTF(validation_loss_str.c_str());
}
