## Getting Started

This is a guide on how to use the resources in this repository to build the android app. There are two parts to performing on device training:
1. Offline processing step - This step builds all the files necessary for performing training.
    
    The files generated in this step are:
    - The training onnx model.
    - The eval onnx model.
    - The optimizer onnx model.
    - The checkpoint file.
    - The data for the model.

    The python file [prepare_android_assets](offline_preprocessing/prepare_android_assets.py) will help prepare all these files. 
    ```py
    python prepare_android_assets.py
    ```
    After running the python file, you should see the `assets` folder in your working directory with the following folder structure
    ```
    assets
        checkpoint
            paramtrain_tensors.pbseq
        test_data
            input_0.bin
            .
            .
            .
            input_9.bin
            labels_0.bin
            .
            .
            .
            labels_9.bin
        train_data
            input_0.bin
            .
            .
            .
            input_49.bin
            labels_0.bin
            .
            .
            .
            labels_49.bin
        adamw_optimizer.onnx
        classifier_eval_model.onnx
        classifier_training_model.onnx
    ```
    We will be needing to copy all these files to the android app later. Keep these generated files aside for the time being. 

2. The actuat training on an android device.

    Follow along the following steps in order to perform the training on the android device:

    - Build onnxruntime-training for on device training. The Ninja generator needs to be used to build on Windows as the Visual Studio generator doesn’t support Android.

        ```sh
        ./build.bat --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)> --cmake_generator Ninja --enable_training --enable_training_on_device --skip_tests --build_shared_lib --config MinSizeRel
        ```

        e.g. using the paths in our example:

        ```sh
        ./build.bat --android --android_sdk_path C:\Users\<username>\AppData\Local\Android\Sdk --android_ndk_path C:\Users\<username\AppData\Local\Android\Sdk\ndk\25.1.8937393 --android_abi arm64-v8a --android_api 27 --cmake_generator  Ninja --enable_training --enable_training_on_device --skip_tests --build_shared_lib --config MinSizeRel
        ```

        We will use the built lib libonnxruntime.so and some of the header files in the android app. Be ready to copy them to a directory later.

    - Copy the lib libonnxruntime.so from the onnxruntime build directory to on_device_training/android_application/app/libs directory
    - Copy the following header files from onnxruntime repo to on_device_training/android_application/app/src/main/cpp/include:
      - include/onnxruntime/core/session/onnxruntime_c_api.h
      - include/onnxruntime/core/session/onnxruntime_cxx_api.h
      - include/onnxruntime/core/session/onnxruntime_cxx_inline.h
      - orttraining/orttraining/training_api/include/onnxruntime_training_c_api.h
      - orttraining/orttraining/training_api/include/onnxruntime_training_cxx_api.h
      - orttraining/orttraining/training_api/include/onnxruntime_training_cxx_inline.h

    - Copy the contents of assets from the offline processing step to on_device_training/android_application/app/src/main/assets

    - Open Android Studio and open the project on_device_training/android_application
    - Build and Run on an android device.
    - Run training on device
