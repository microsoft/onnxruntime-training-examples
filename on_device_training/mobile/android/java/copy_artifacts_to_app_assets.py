# copies over training artifacts and training data to the Android app assets directory

import pathlib
import shutil


script_dir = pathlib.Path(__file__).parent
app_assets_dir = pathlib.Path(script_dir, "app/ORTPersonalize/app/src/main/assets")

# copy over training artifacts
training_artifacts_dir = script_dir / "training_artifacts"

if len(list(training_artifacts_dir.glob("*.onnx"))) == 0:
    raise RuntimeError("Use prepare_for_training.ipynb to generate the training artifacts first.")

shutil.copytree(training_artifacts_dir, app_assets_dir / "training_artifacts", dirs_exist_ok=True)


# copy over training image data
training_image_data_dir = script_dir / "data" / "images"

shutil.copytree(training_image_data_dir, app_assets_dir / "images", dirs_exist_ok=True)
