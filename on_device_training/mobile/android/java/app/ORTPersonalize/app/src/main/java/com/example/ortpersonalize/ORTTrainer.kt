package com.example.ortpersonalize

import ai.onnxruntime.*
import android.R.attr.key
import android.R.attr.value
import android.util.Log
import androidx.annotation.InspectableProperty.ValueType
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Collections
import kotlin.math.exp


class ORTTrainer{
    private var ortEnv: OrtEnvironment? = null
    private var ortTrainingSession: OrtTrainingSession? = null
    private var ortSession: OrtSession? = null

    constructor(checkpointPath: String, trainModelPath: String, evalModelPath: String, optimizerModelPath: String) {
        ortEnv = OrtEnvironment.getEnvironment()
        ortTrainingSession = ortEnv?.createTrainingSession(checkpointPath, trainModelPath, evalModelPath, optimizerModelPath)
    }

    public fun performTraining(data: FloatBuffer, labels: LongBuffer, batchSize: Long) {
        ortSession = null

        var loss = -1.0f
        ortEnv.use {
            val dataShape = longArrayOf(batchSize, 3, 224, 224)
            val inputTensor = OnnxTensor.createTensor(ortEnv, data, dataShape)
            val labelsShape = longArrayOf(batchSize)
            val labelsTensor = OnnxTensor.createTensor(ortEnv, labels, labelsShape)
            inputTensor.use {
                labelsTensor.use {
                    val ortInputMap: MutableMap<String, OnnxTensor> = HashMap<String, OnnxTensor>()
                    ortInputMap["input"] = inputTensor
                    ortInputMap["labels"] = labelsTensor
                    val output = ortTrainingSession?.trainStep(ortInputMap)
                    output.use {
                        @Suppress("UNCHECKED_CAST")
                        loss = ((output?.get(0)?.value) as Float)
                    }
                }
            }

            ortTrainingSession?.optimizerStep()
            ortTrainingSession?.lazyResetGrad()
        }
    }

    public fun performInference(imgData: FloatBuffer, classes: Array<String>, cacheDir: File): String {
        if (ortSession == null) {
            val inferenceModelPath: Path = Paths.get(cacheDir.toString(), "inference_model.onnx")
            val graphOutput: Array<String> = arrayOf("output")
            ortTrainingSession?.exportModelForInference(inferenceModelPath, graphOutput)
            ortSession = ortEnv?.createSession(inferenceModelPath.toString())
        }
        var maxIdx = -1
        ortEnv.use {
            val shape = longArrayOf(1, 3, 224, 224)
            val tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
            tensor.use {
                val output = ortSession?.run(Collections.singletonMap("input", tensor))
                output.use {
                    @Suppress("UNCHECKED_CAST")
                    val rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                    val probabilities = softMax(rawOutput)

                    maxIdx = probabilities.indices.maxBy { probabilities[it] } ?: -1
                }
            }
        }

        check(maxIdx >= 0) { "Index is < 0" }

        return classes[maxIdx]
    }

    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }
}
