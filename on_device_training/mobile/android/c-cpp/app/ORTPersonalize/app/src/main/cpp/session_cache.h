//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_SESSION_CACHE_H
#define ORT_PERSONALIZE_SESSION_CACHE_H

#include "onnxruntime_training_cxx_api.h"

struct ArtifactPaths {
    std::string checkpoint_path;
    std::string training_model_path;
    std::string eval_model_path;
    std::string optimizer_model_path;
    std::string cache_dir_path;
    std::string inference_model_path;

    ArtifactPaths(const std::string &checkpoint_path, const std::string &training_model_path,
                  const std::string &eval_model_path, const std::string &optimizer_model_path,
                  const std::string& cache_dir_path) :
            checkpoint_path(checkpoint_path), training_model_path(training_model_path),
            eval_model_path(eval_model_path), optimizer_model_path(optimizer_model_path),
            cache_dir_path(cache_dir_path), inference_model_path(cache_dir_path + "/inference.onnx") {}
};

struct SessionCache {
    ArtifactPaths artifact_paths;
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    Ort::CheckpointState checkpoint_state;
    Ort::TrainingSession training_session;
    Ort::Session* inference_session;

    SessionCache(const std::string &checkpoint_path, const std::string &training_model_path,
                 const std::string &eval_model_path, const std::string &optimizer_model_path,
                 const std::string& cache_dir_path) :
            artifact_paths(checkpoint_path, training_model_path, eval_model_path, optimizer_model_path, cache_dir_path),
            ort_env(ORT_LOGGING_LEVEL_WARNING, "ort personalize"), session_options(),
            checkpoint_state(Ort::CheckpointState::LoadCheckpoint(artifact_paths.checkpoint_path.c_str())),
            training_session(ort_env, session_options, checkpoint_state, artifact_paths.training_model_path.c_str(),
                             artifact_paths.eval_model_path.c_str(), artifact_paths.optimizer_model_path.c_str()),
            inference_session(nullptr) {}
};

#endif //ORT_PERSONALIZE_SESSION_CACHE_H