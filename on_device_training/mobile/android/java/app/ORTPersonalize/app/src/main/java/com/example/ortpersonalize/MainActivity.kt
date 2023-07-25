package com.example.ortpersonalize

import ai.onnxruntime.*
import android.Manifest
import android.app.Dialog
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.text.InputType
import android.util.Log
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.example.ortpersonalize.databinding.ActivityMainBinding
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.*


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var ort_session: Long = -1 // Give a default initial value.
    private val PICK_IMAGE = 1000 // Pick an image for inferencing from the android gallery
    private val CAPTURE_IMAGE = 2000 // Capture an image from the camera
    private val PICK_CLASS_A_IMAGES_FOR_TRAINING =
        4000 // Pick class A images for training from the android gallery
    private val PICK_CLASS_B_IMAGES_FOR_TRAINING =
        5000 // Pick class B images for training from the android gallery
    private val PICK_CLASS_X_IMAGES_FOR_TRAINING =
        6000 // Pick class X images for training from the android gallery
    private val PICK_CLASS_Y_IMAGES_FOR_TRAINING =
        7000 // Pick class Y images for training from the android gallery
    private val CAMERA_PERMISSION_CODE = 8 // Permission to access the camera
    private var images =
        ArrayList<Pair<Uri, Int>>() // Array that stores the Uri for all images that need to be trained.
    private var samplesClassA = 0 // Number of samples for class A
    private var nameClassA: String = "A" // Default class A name
    private var samplesClassB = 0 // Number of samples for class B
    private var nameClassB: String = "B" // Default class B name
    private var samplesClassX = 0 // Number of samples for class X
    private var nameClassX: String = "X" // Default class X name
    private var samplesClassY = 0 // Number of samples for class Y
    private var nameClassY: String = "Y" // Default class Y name
    private val prepackedDefaultLabels: Array<String> =
        arrayOf("dog", "cat", "elephant", "cow") // Default labels for non custom class
    private var ortTrainer: ORTTrainer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val trainingModelPath =
            copyAssetToCacheDir("mobilenetv2_training.onnx", "training_model.onnx")
        val evalModelPath = copyAssetToCacheDir("mobilenetv2_eval.onnx", "eval_model.onnx")
        val checkpointPath = copyFileOrDir("mobilenetv2.ckpt")
        val optimizerModelPath =
            copyAssetToCacheDir("mobilenetv2_optimizer.onnx", "optimizer_model.onnx")
        ortTrainer =
            ORTTrainer(checkpointPath, trainingModelPath, evalModelPath, optimizerModelPath)

        val inferButton: Button = findViewById(R.id.infer_button)
        inferButton.setOnClickListener(onInferenceButtonClickedListener)

        val trainButton: Button = findViewById(R.id.train_button)
        trainButton.setOnClickListener(onTrainButtonClickedListener)
        trainButton.isEnabled = false

        val classA: Button = findViewById(R.id.classA)
        classA.setOnClickListener(onClassAClickedListener)
        classA.setOnLongClickListener(onClassALongClickedListener)

        val classB: Button = findViewById(R.id.classB)
        classB.setOnClickListener(onClassBClickedListener)
        classB.setOnLongClickListener(onClassBLongClickedListener)

        val classX: Button = findViewById(R.id.classX)
        classX.setOnClickListener(onClassXClickedListener)
        classX.setOnLongClickListener(onClassXLongClickedListener)

        val classY: Button = findViewById(R.id.classY)
        classY.setOnClickListener(onClassYClickedListener)
        classY.setOnLongClickListener(onClassYLongClickedListener)

        binding.customClassSetting.setOnCheckedChangeListener(onCustomClassSettingChangedListener)

        // Home screen
        binding.statusMessage.text = "ORT Personalize"
    }

    private val onInferenceButtonClickedListener: View.OnClickListener =
        object : View.OnClickListener {
            override fun onClick(v: View) {
                val cameraSetting: Switch = findViewById(R.id.camera_setting)
                if (cameraSetting.isChecked()) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                        requestPermissions(
                            arrayOf(Manifest.permission.CAMERA),
                            CAMERA_PERMISSION_CODE
                        )
                    } else {
                        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                        startActivityForResult(cameraIntent, CAPTURE_IMAGE)
                    }
                } else {
                    val intent = Intent()
                    intent.type = "image/*"
                    intent.action = Intent.ACTION_GET_CONTENT
                    startActivityForResult(
                        Intent.createChooser(intent, "Select Picture"),
                        PICK_IMAGE
                    )
                }
            }
        }

    private fun disableButtons() {
        binding.classA.isEnabled = false
        binding.classB.isEnabled = false
        binding.classX.isEnabled = false
        binding.classY.isEnabled = false
        binding.trainButton.isEnabled = false
        binding.inferButton.isEnabled = false
    }

    private fun enableButtons() {
        if (binding.customClassSetting.isChecked) {
            binding.classA.isEnabled = true
            binding.classB.isEnabled = true
            binding.classX.isEnabled = true
            binding.classY.isEnabled = true
        }
        binding.trainButton.isEnabled = false
        binding.inferButton.isEnabled = true
    }

    private val onTrainButtonClickedListener: View.OnClickListener = object : View.OnClickListener {
        override fun onClick(v: View) {
            // Reset the samples
            samplesClassA = 0
            samplesClassB = 0
            samplesClassX = 0
            samplesClassY = 0

            // Reset the state message
            binding.statusMessage.text = ""
            // Reset the image view
            binding.inputImage.setImageResource(0)

            // Update the class names based on whether customClassSetting is checked or not
            if (binding.customClassSetting.isChecked) {
                binding.classA.text = nameClassA
                binding.classB.text = nameClassB
                binding.classX.text = nameClassX
                binding.classY.text = nameClassY
            } else {
                binding.classA.text = prepackedDefaultLabels[0]
                binding.classB.text = prepackedDefaultLabels[1]
                binding.classX.text = prepackedDefaultLabels[2]
                binding.classY.text = prepackedDefaultLabels[3]

                // Move asset images to the cache and collect their Uris for training.
                val cachePath: String = copyFileOrDir("images")
                for ((index, label) in prepackedDefaultLabels.withIndex()) {
                    for (image_num in 1..20) {
                        val imagePath: String =
                            String.format("%s/%s/%s%02d.jpeg", cachePath, label, label, image_num)
                        val f = java.io.File(imagePath)
                        images.add(Pair(Uri.fromFile(f), index))
                    }
                }
            }
            binding.trainButton.isEnabled = false

            disableButtons()
            binding.statusMessage.text = ""

            val dialog = Dialog(v.context)
            dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
            dialog.setTitle("Training...")
            dialog.setCancelable(false)
            dialog.setContentView(R.layout.dialog)

            val text = dialog.findViewById(R.id.progress_horizontal) as ProgressBar
            val percentage: TextView = dialog.findViewById(R.id.percent_complete)
            val trainingStatus: TextView = dialog.findViewById(R.id.train_status)
            trainingStatus.text = "Training... (epoch: 0/5)"

            dialog.show()
            val window = dialog.window
            window!!.setLayout(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT
            )

            Thread(Runnable {
                val batchSize: Int = 4
                val channels: Int = 3
                val width: Int = 224
                val height: Int = 224
                val numEpochs: Int = 5

                for (epoch in 0 until numEpochs) {
                    // Shuffle the images so that we don't have a bias during training
                    Collections.shuffle(images);
                    for (i in 0..images.size - 1 step batchSize) {
                        val imgData = FloatBuffer.allocate(batchSize * channels * width * height)
                        imgData.rewind()
                        val labels = IntBuffer.allocate(batchSize)
                        labels.rewind()
                        for (j in 0 until batchSize) {
                            if (i + j >= images.size) {
                                break
                            }

                            labels.put(j, images[i + j].second)

                            val bitmap: Bitmap =
                                processBitmap(bitmapFromUri(images[i + j].first, contentResolver))
                            val imageWidth: Int = bitmap.getWidth()
                            val imageHeight: Int = bitmap.getHeight()
                            val stride: Int = imageWidth * imageHeight
                            val offset = j * stride * channels

                            processImage(bitmap, imgData, offset)
                        }
                        imgData.rewind()
                        labels.rewind()

                        ortTrainer?.performTraining(
                            imgData, labels, batchSize.toLong()
                        )

                        this@MainActivity.runOnUiThread(java.lang.Runnable {
                            val status: Int =
                                (100f * ((epoch * images.size).toFloat() + i.toFloat()) / (images.size * numEpochs)).toInt()
                            text.setProgress(status)
                            percentage.setText(status.toString())
                        })
                    }
                    this@MainActivity.runOnUiThread(java.lang.Runnable {
                        trainingStatus.setText(
                            String.format(
                                "Training... (epoch: %d/%d)",
                                epoch + 1,
                                numEpochs
                            )
                        )
                    })
                }

                this@MainActivity.runOnUiThread(java.lang.Runnable {
                    images.clear()
                    dialog.dismiss()
                    binding.statusMessage.text = "Training Complete"
                    enableButtons()
                })
            }).start()
        }
    }

    private val onCustomClassSettingChangedListener: CompoundButton.OnCheckedChangeListener =
        object : CompoundButton.OnCheckedChangeListener {
            override fun onCheckedChanged(button: CompoundButton?, isChecked: Boolean) {
                disableButtons()
                binding.statusMessage.text = ""
                binding.inputImage.setImageResource(0)
                if (isChecked) {
                    binding.classA.text = nameClassA
                    binding.classB.text = nameClassB
                    binding.classX.text = nameClassX
                    binding.classY.text = nameClassY
                    ortTrainer = null
                    val trainingModelPath =
                        copyAssetToCacheDir("mobilenetv2_training.onnx", "training_model.onnx")
                    val evalModelPath =
                        copyAssetToCacheDir("mobilenetv2_eval.onnx", "eval_model.onnx")
                    val checkpointPath = copyFileOrDir("mobilenetv2.ckpt")
                    val optimizerModelPath =
                        copyAssetToCacheDir("mobilenetv2_optimizer.onnx", "optimizer_model.onnx")
                    ortTrainer = ORTTrainer(checkpointPath, trainingModelPath, evalModelPath, optimizerModelPath)
                    enableButtons()
                } else {
                    binding.classA.text = String.format("%s (20)", prepackedDefaultLabels[0])
                    binding.classB.text = String.format("%s (20)", prepackedDefaultLabels[1])
                    binding.classX.text = String.format("%s (20)", prepackedDefaultLabels[2])
                    binding.classY.text = String.format("%s (20)", prepackedDefaultLabels[3])
                    ortTrainer = null
                    val trainingModelPath =
                        copyAssetToCacheDir("mobilenetv2_training.onnx", "training_model.onnx")
                    val evalModelPath =
                        copyAssetToCacheDir("mobilenetv2_eval.onnx", "eval_model.onnx")
                    val checkpointPath = copyFileOrDir("mobilenetv2.ckpt")
                    val optimizerModelPath =
                        copyAssetToCacheDir("mobilenetv2_optimizer.onnx", "optimizer_model.onnx")
                    ortTrainer = ORTTrainer(checkpointPath, trainingModelPath, evalModelPath, optimizerModelPath)
                    binding.trainButton.isEnabled = true
                    binding.inferButton.isEnabled = true
                }
            }
        }

    private val onClassAClickedListener: View.OnClickListener = object : View.OnClickListener {
        override fun onClick(v: View) {
            val intent = Intent()
            intent.type = "image/*"
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(
                Intent.createChooser(intent, "Select Picture"),
                PICK_CLASS_A_IMAGES_FOR_TRAINING
            )
        }
    }

    private val onClassALongClickedListener: View.OnLongClickListener =
        object : View.OnLongClickListener {
            override fun onLongClick(v: View): Boolean {
                val builder: AlertDialog.Builder = AlertDialog.Builder(v.context)
                builder.setTitle("Change class name")

                val input = EditText(v.context)
                input.inputType = InputType.TYPE_CLASS_TEXT
                builder.setView(input)

                builder.setPositiveButton("OK",
                    DialogInterface.OnClickListener { dialog, which ->
                        val classA: Button = findViewById(R.id.classA)
                        nameClassA = input.text.toString()
                        classA.text = nameClassA
                    })
                builder.setNegativeButton("Cancel",
                    DialogInterface.OnClickListener { dialog, which -> dialog.cancel() })

                builder.show()

                return true
            }
        }

    private val onClassBClickedListener: View.OnClickListener = object : View.OnClickListener {
        override fun onClick(v: View) {
            val intent = Intent()
            intent.type = "image/*"
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(
                Intent.createChooser(intent, "Select Picture"),
                PICK_CLASS_B_IMAGES_FOR_TRAINING
            )
        }
    }

    private val onClassBLongClickedListener: View.OnLongClickListener =
        object : View.OnLongClickListener {
            override fun onLongClick(v: View): Boolean {
                val builder: AlertDialog.Builder = AlertDialog.Builder(v.context)
                builder.setTitle("Change class name")

                val input = EditText(v.context)
                input.inputType = InputType.TYPE_CLASS_TEXT
                builder.setView(input)

                builder.setPositiveButton("OK",
                    DialogInterface.OnClickListener { dialog, which ->
                        val classB: Button = findViewById(R.id.classB)
                        nameClassB = input.text.toString()
                        classB.text = nameClassB
                    })
                builder.setNegativeButton("Cancel",
                    DialogInterface.OnClickListener { dialog, which -> dialog.cancel() })

                builder.show()

                return true
            }
        }

    private val onClassXClickedListener: View.OnClickListener = object : View.OnClickListener {
        override fun onClick(v: View) {
            val intent = Intent()
            intent.type = "image/*"
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(
                Intent.createChooser(intent, "Select Picture"),
                PICK_CLASS_X_IMAGES_FOR_TRAINING
            )
        }
    }

    private val onClassXLongClickedListener: View.OnLongClickListener =
        object : View.OnLongClickListener {
            override fun onLongClick(v: View): Boolean {
                val builder: AlertDialog.Builder = AlertDialog.Builder(v.context)
                builder.setTitle("Change class name")

                val input = EditText(v.context)
                input.inputType = InputType.TYPE_CLASS_TEXT
                builder.setView(input)

                builder.setPositiveButton("OK",
                    DialogInterface.OnClickListener { dialog, which ->
                        val classX: Button = findViewById(R.id.classX)
                        nameClassX = input.text.toString()
                        classX.text = nameClassX
                    })
                builder.setNegativeButton("Cancel",
                    DialogInterface.OnClickListener { dialog, which -> dialog.cancel() })

                builder.show()

                return true
            }
        }

    private val onClassYClickedListener: View.OnClickListener = object : View.OnClickListener {
        override fun onClick(v: View) {
            val intent = Intent()
            intent.type = "image/*"
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(
                Intent.createChooser(intent, "Select Picture"),
                PICK_CLASS_Y_IMAGES_FOR_TRAINING
            )
        }
    }

    private val onClassYLongClickedListener: View.OnLongClickListener =
        object : View.OnLongClickListener {
            override fun onLongClick(v: View): Boolean {
                val builder: AlertDialog.Builder = AlertDialog.Builder(v.context)
                builder.setTitle("Change class name")

                val input = EditText(v.context)
                input.inputType = InputType.TYPE_CLASS_TEXT
                builder.setView(input)

                builder.setPositiveButton("OK",
                    DialogInterface.OnClickListener { dialog, which ->
                        val classY: Button = findViewById(R.id.classY)
                        nameClassY = input.text.toString()
                        classY.text = nameClassY
                    })
                builder.setNegativeButton("Cancel",
                    DialogInterface.OnClickListener { dialog, which -> dialog.cancel() })

                builder.show()

                return true
            }
        }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String?>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show()
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, CAPTURE_IMAGE)
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != RESULT_OK) {
            return
        }

        lateinit var srcBitMap: Bitmap
        if (requestCode == PICK_IMAGE) {
            srcBitMap = data?.data?.let { bitmapFromUri(it, contentResolver) }!!
        } else if (requestCode == CAPTURE_IMAGE) {
            srcBitMap = data?.extras?.get("data") as Bitmap
        } else if (requestCode == PICK_CLASS_A_IMAGES_FOR_TRAINING) {
            if (data?.getClipData() != null) {
                val count: Int = data?.getClipData()!!.getItemCount()
                for (i in 0 until count) {
                    samplesClassA += 1
                    val pair = Pair(data?.getClipData()!!.getItemAt(i).getUri(), 0)
                    images.add(pair)
                }
            } else if (data?.getData() != null) {
                samplesClassA += 1
                val pair = Pair(data?.getData()!!, 0);
                images.add(pair)
            }
            binding.classA.text = String.format("%s (%d)", nameClassA, samplesClassA)
            binding.trainButton.isEnabled = true
            binding.statusMessage.text = ""
            binding.inputImage.setImageResource(0)
            return
        } else if (requestCode == PICK_CLASS_B_IMAGES_FOR_TRAINING) {
            if (data?.getClipData() != null) {
                val count: Int = data?.getClipData()!!.getItemCount()
                for (i in 0 until count) {
                    samplesClassB += 1
                    val pair = Pair(data?.getClipData()!!.getItemAt(i).getUri(), 1)
                    images.add(pair)
                }
            } else if (data?.getData() != null) {
                samplesClassB += 1
                val pair = Pair(data.getData()!!, 1);
                images.add(pair)
            }
            binding.classB.text = String.format("%s (%d)", nameClassB, samplesClassB)
            binding.trainButton.isEnabled = true
            binding.statusMessage.text = ""
            binding.inputImage.setImageResource(0)
            return
        } else if (requestCode == PICK_CLASS_X_IMAGES_FOR_TRAINING) {
            if (data?.getClipData() != null) {
                val count: Int = data?.getClipData()!!.getItemCount()
                for (i in 0 until count) {
                    samplesClassX += 1
                    val pair = Pair(data?.getClipData()!!.getItemAt(i).getUri(), 2)
                    images.add(pair)
                }
            } else if (data?.getData() != null) {
                samplesClassX += 1
                val pair = Pair(data?.getData()!!, 2);
                images.add(pair)
            }
            binding.classX.text = String.format("%s (%d)", nameClassX, samplesClassX)
            binding.trainButton.isEnabled = true
            binding.statusMessage.text = ""
            binding.inputImage.setImageResource(0)
            return
        } else if (requestCode == PICK_CLASS_Y_IMAGES_FOR_TRAINING) {
            if (data?.getClipData() != null) {
                val count: Int = data?.getClipData()!!.getItemCount()
                for (i in 0 until count) {
                    samplesClassY += 1
                    val pair = Pair(data?.getClipData()!!.getItemAt(i).getUri(), 3)
                    images.add(pair)
                }
            } else if (data?.getData() != null) {
                samplesClassY += 1
                val pair = Pair(data?.getData()!!, 3);
                images.add(pair)
            }
            binding.classY.text = String.format("%s (%d)", nameClassY, samplesClassY)
            binding.trainButton.isEnabled = true
            binding.statusMessage.text = ""
            binding.inputImage.setImageResource(0)
            return
        }

        if (srcBitMap != null) {
            val batchSize: Int = 1
            val channels: Int = 3
            val width: Int = 224
            val height: Int = 224

            val bitmapResized: Bitmap = processBitmap(srcBitMap)

            binding.inputImage.setImageBitmap(bitmapResized)

            val imgData = FloatBuffer.allocate(batchSize * channels * width * height)
            imgData.rewind()
            processImage(bitmapResized, imgData, 0)
            imgData.rewind()
            var classes: Array<String>
            if (binding.customClassSetting.isChecked) {
                classes = arrayOf(nameClassA, nameClassB, nameClassX, nameClassY)
            } else {
                classes = prepackedDefaultLabels
            }
            binding.statusMessage.text = ortTrainer?.let {
                String.format(
                    "Prediction: %s",
                    it.performInference(imgData, classes, cacheDir)
                )
            }

            if (requestCode == CAPTURE_IMAGE) {
                binding.cameraSetting.isChecked = true
            }
        }
    }

    private fun mkCacheDir(cacheFileName: String) {
        val dirs = cacheFileName.split("/")
        var extendedCacheDir = "$cacheDir"
        for (index in 0..dirs.size - 2) {
            val myDir = java.io.File(extendedCacheDir, dirs.get(index))
            myDir.mkdir()
            extendedCacheDir = extendedCacheDir + "/" + dirs.get(index)
        }
    }

    // copy file from asset to cache dir in the same dir structure.
    private fun copyAssetToCacheDir(assetFileName: String, cacheFileName: String): String {
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
            Log.e("ortpersonalize", "I/O Exception", ex)
        }

        return "$cacheDir/$path"
    }
}