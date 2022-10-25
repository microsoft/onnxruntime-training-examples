package com.example.ondevicetraining

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.ondevicetraining.databinding.ActivityMainBinding
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.util.Log
import android.view.View
import androidx.annotation.RequiresApi
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val REQUEST_PICK_IMAGE = 1000
    private var trainingResource: Long? = null
    private var trainingStep: Int = 0
    private val totalTrainingSteps = 50

    // Instantiate the training session and display welcome message to users on opening the app
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (trainingResource == null) {
            val checkpointPath = copyFileOrDir("checkpoint")
            val trainingModelPath = copyAssetToCacheDir("classifier_training_model.onnx", "training_model.onnx")
            val evalModelPath = copyAssetToCacheDir("classifier_eval_model.onnx", "eval_model.onnx")
            val optimizerModelPath = copyAssetToCacheDir("adamw_optimizer.onnx", "optimizer_model.onnx")

            trainingResource = getTrainingSessionCache(checkpointPath, trainingModelPath, evalModelPath, optimizerModelPath, totalTrainingSteps)
        }


        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val welcomeMessage = "onnxruntime on-device training"
        binding.sampleText.text = welcomeMessage
    }

    // Release any held training resources when app is closed.
    override fun onDestroy() {
        super.onDestroy()

        trainingResource?.let { releaseTrainingResource(it) }
    }

    // Helper functions

    // create dir structure inside cache based on file name
    private fun mkCacheDir(cacheFileName: String) {
        val dirs = cacheFileName.split("/")
        var extendedCacheDir = "$cacheDir"
        for (index in 0..dirs.size-2) {
            val myDir = java.io.File(extendedCacheDir, dirs.get(index))
            myDir.mkdir()
            extendedCacheDir = extendedCacheDir + dirs.get(index)
        }
    }

    // copy file from asset to cache dir in the same dir structure.
    private fun copyAssetToCacheDir(assetFileName : String, cacheFileName : String): String {
        mkCacheDir(cacheFileName)
        val f = java.io.File("$cacheDir/$cacheFileName")
        if (!f.exists()) {
            try {
                val modelFile = assets.open(assetFileName)
                val size: Int = modelFile.available()
                val buffer = ByteArray(size)
                modelFile.read(buffer)
                modelFile.close()
                val fos = java.io.FileOutputStream(f)
                fos.write(buffer)
                fos.close()
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
        }

        return f.path
    }

    private fun copyFileOrDir(path: String): String {
        val assetManager = assets
        try {
            val assets: Array<String>? = assetManager.list(path)
            if (assets!!.size == 0) {
                // asset is a file
                copyAssetToCacheDir(path, path)
            } else {
                // asset is a dir. loop over dir and copy all files or sub dirs to cache dir
                for (i in assets.indices) {
                    var p: String
                    p = if (path == "") "" else "$path/"
                    copyFileOrDir(p + assets[i])
                }
            }
        } catch (ex: java.io.IOException) {
            Log.e("ondevicetraining", "I/O Exception", ex)
        }

        return "$cacheDir/$path"
    }


    /**
     * A native method that is implemented by the 'ondevicetraining' native library,
     * which is packaged with this application.
     */

    // Training session cache contains all objects necessary for performing training
    external fun getTrainingSessionCache(
        checkpointPath:String,
        trainModelPath: String,
        evalModelPath: String,
        optimizerModelPath: String,
        dataloader_max_steps: Int
    ): Long

    // Releases created training session cache.
    external fun releaseTrainingResource(
        trainingResource: Long
    )

    // Performs inference and returns a class label.
    external fun infer(
        modelPath: String,
        imageBuffer: FloatArray,
        batchSize: Int,
        channels: Int,
        frameCols: Int,
        frameRows: Int,
        trainingResource: Long
    ): String

    // Perform 1 training step
    external fun train(
        trainingResource: Long,
        trainDataSetCacheDir: String
    ): String

    // Perform eval on test data
    external fun eval(
        trainingResource: Long,
        testDataSetCacheDir: String
    ): String

    companion object {
        // Used to load the 'ondevicetraining' library on application startup.
        init {
            System.loadLibrary("ondevicetraining")
        }
    }

    // Inference related functions

    // Given an image resource, convert to bitmap and return
    @RequiresApi(Build.VERSION_CODES.P)
    private fun createBitmapFromUri(uri: Uri): Bitmap {
        val source: ImageDecoder.Source = ImageDecoder.createSource(contentResolver, uri)
        return ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.ARGB_8888, true)
    }

    // Called when UI action for inference is selected.
    fun onPickImage(view: View) {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), REQUEST_PICK_IMAGE)
    }

    @Deprecated("Deprecated in Java")
    @RequiresApi(Build.VERSION_CODES.P)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != RESULT_OK) {
            return
        }

        if (requestCode == REQUEST_PICK_IMAGE) {
            val srcBitMap: Bitmap? = data?.data?.let { createBitmapFromUri(it) }
            if (srcBitMap != null) {
                performInference(srcBitMap)
            }
        }
    }

    private fun performInference(srcBitMap: Bitmap) {
        // cifar accepts 3x32x32 tensors. We need to crop and resize the image
        val (frameRows, frameCols) = Pair(32, 32)
        val channels = 3
        val batchSize = 1

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Crop the image to be a square (i.e. 3xsxs where s is min(img_width, img_height))
        binding.activityImageViewer.setImageBitmap(srcBitMap)
        lateinit var croppedBitMap: Bitmap
        if (srcBitMap.getWidth() >= srcBitMap.getHeight()) {
            croppedBitMap = Bitmap.createBitmap(
                srcBitMap,
                srcBitMap.getWidth() / 2 - srcBitMap.getHeight() / 2,
                0,
                srcBitMap.getHeight(),
                srcBitMap.getHeight()
            )
        } else {
            croppedBitMap = Bitmap.createBitmap(
                srcBitMap,
                0,
                srcBitMap.getHeight() / 2 - srcBitMap.getWidth() / 2,
                srcBitMap.getWidth(),
                srcBitMap.getWidth()
            )
        }

        // Resize the image to be 3x32x32
        val dstBitMap = Bitmap.createScaledBitmap(
            croppedBitMap, frameCols, frameRows, false
        )

        binding.resizedImageViewer.setImageBitmap(dstBitMap)

        // Extract float buffer from image and preprocess it by normalization:
        // float array is in range [0, 1]
        // mean (0.5, 0.5, 0.5) is subtracted from float array
        // float array is divided by std (0.5, 0.5, 0.5).
        // The tuple represents the 3 channels.
        val imgData = FloatBuffer.allocate(batchSize * channels * frameCols * frameCols)
        imgData.rewind()
        val imageWidth: Int = dstBitMap.getWidth()
        val imageHeight: Int = dstBitMap.getHeight()
        val stride: Int = imageWidth * imageHeight
        for (row in 0..imageHeight-1) {
            for (col in 0..imageWidth-1) {
                val color: Int = dstBitMap.getPixel(col, row)
                // Range [0, 1]
                val alpha = Color.alpha(color).toFloat() / 255
                val red = Color.red(color).toFloat() / 255
                val green = Color.green(color).toFloat() / 255
                val blue = Color.blue(color).toFloat() / 255
                val index = row * imageWidth + col
                // Subtract mean and divide by std
                imgData.put(index, ((alpha * red) - 0.5f) / 0.5f)
                imgData.put(index + stride, ((alpha * green) - 0.5f) / 0.5f)
                imgData.put(index + stride + stride, ((alpha * blue)  - 0.5f) / 0.5f)
            }
        }
        imgData.rewind()

        // Perform inference and get the prediction.
        val predictionStr = "That's " + trainingResource?.let {
            infer("$cacheDir/inference_model.onnx", imgData.array(),
                batchSize, channels, frameCols, frameRows, it)
        }
        binding.sampleText.text = predictionStr
    }

    // Training related functions

    fun performTraining(view: View) {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val trainDataSetCacheDir = copyFileOrDir("train_data")


        // Example of a call to a native method
        binding.durationText.text = trainingResource?.let { train(it, trainDataSetCacheDir) }

        trainingStep++
        binding.trainingProgressBar.setProgress((trainingStep * 100) / totalTrainingSteps)
        if (trainingStep < totalTrainingSteps) {
            val trainingStatus = "Training in Progress " + ((trainingStep * 100) / totalTrainingSteps).toString() + "%."
            binding.sampleText.text = trainingStatus
        }
        else {
            trainingStep = totalTrainingSteps
            val trainingStatus = "Training Complete."
            binding.sampleText.text = trainingStatus
        }
    }

    fun performEval(view: View) {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val evalDataSetCacheDir = copyFileOrDir("test_data")

        binding.trainingProgressBar.setProgress((trainingStep * 100) / totalTrainingSteps)
        if (trainingStep < totalTrainingSteps) {
            val trainingStatus = "Training in Progress " + ((trainingStep * 100) / totalTrainingSteps).toString() + "%."
            binding.sampleText.text = trainingStatus
        }
        else {
            trainingStep = totalTrainingSteps
            val trainingStatus = "Training Complete."
            binding.sampleText.text = trainingStatus
        }
        binding.durationText.text = trainingResource?.let {
            eval(it,
                evalDataSetCacheDir
            )
        }
    }
}