#include <jni.h>
#include <string>
#include <cassert>
#include <memory>

#include "onnxruntime_training_cxx_api.h"
#include "inference.h"
#include "session_cache.h"
#include "utils.h"
#include "train.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_ortpersonalize_MainActivity_getSession(
        JNIEnv *env, jobject /* this */,
        jstring checkpoint_path, jstring train_model_path, jstring eval_model_path,
        jstring optimizer_model_path, jstring cache_dir_path)
{
    std::unique_ptr<SessionCache> session_cache = std::make_unique<SessionCache>(
            utils::JString2String(env, checkpoint_path),
            utils::JString2String(env, train_model_path),
            utils::JString2String(env, eval_model_path),
            utils::JString2String(env, optimizer_model_path),
            utils::JString2String(env, cache_dir_path));
    return reinterpret_cast<long>(session_cache.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ortpersonalize_MainActivity_releaseSession(
        JNIEnv *env, jobject /* this */,
        jlong session) {
    auto *session_cache = reinterpret_cast<SessionCache *>(session);
    delete session_cache->inference_session;
    delete session_cache;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ortpersonalize_MainActivity_performInference(
        JNIEnv *env, jobject  /* this */,
        jlong session, jfloatArray image_buffer, jint batch_size, jint image_channels, jint image_rows,
        jint image_cols, jobjectArray classes) {

    std::vector<std::string> classes_str;
    for (int i = 0; i < env->GetArrayLength(classes); ++i) {
        // access the current string element
        jstring elem = static_cast<jstring>(env->GetObjectArrayElement(classes, i));
        classes_str.push_back(utils::JString2String(env, elem));
    }

    auto* session_cache = reinterpret_cast<SessionCache *>(session);
    if (!session_cache->inference_session) {
        // the inference session does not exist, so create a new one.
        session_cache->training_session.ExportModelForInferencing(
                session_cache->artifact_paths.inference_model_path.c_str(), {"output"});
        session_cache->inference_session = std::make_unique<Ort::Session>(
                session_cache->ort_env, session_cache->artifact_paths.inference_model_path.c_str(),
                session_cache->session_options).release();
    }

    auto prediction = inference::classify(
            session_cache, env->GetFloatArrayElements(image_buffer, nullptr),
            batch_size, image_channels, image_rows, image_cols, classes_str);

    return env->NewStringUTF(prediction.first.c_str());
}
extern "C"
JNIEXPORT float JNICALL
Java_com_example_ortpersonalize_MainActivity_performTraining(
        JNIEnv *env, jobject /* this */,
        jlong session, jfloatArray batches, jintArray labels, jint batch_size,
        jint channels, jint frame_rows, jint frame_cols) {
    auto* session_cache = reinterpret_cast<SessionCache *>(session);

    if (session_cache->inference_session) {
        // invalidate the inference session since we will be updating the model parameters
        // in train_step.
        // the next call to inference session will need to recreate the inference session.
        delete session_cache->inference_session;
        session_cache->inference_session = nullptr;
    }

    // update the model parameters using this batch of inputs.
    return training::train_step(session_cache, env->GetFloatArrayElements(batches, nullptr),
                                env->GetIntArrayElements(labels, nullptr), batch_size,
                                channels, frame_rows, frame_cols);
}