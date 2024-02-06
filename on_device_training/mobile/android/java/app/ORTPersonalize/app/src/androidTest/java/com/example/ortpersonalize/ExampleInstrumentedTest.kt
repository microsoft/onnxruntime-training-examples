package com.example.ortpersonalize

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.ortpersonalize", appContext.packageName)
    }

    @Test
    fun createTrainer() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        fun copyFileOrDir(path: String): String {
            val dst = java.io.File("${appContext.cacheDir}/$path")
            copyAssetFileOrDir(appContext.assets, path, dst)
            return dst.path
        }

        val trainingModelPath = copyFileOrDir("training_artifacts/training_model.onnx")
        val evalModelPath = copyFileOrDir("training_artifacts/eval_model.onnx")
        val checkpointPath = copyFileOrDir("training_artifacts/checkpoint")
        val optimizerModelPath = copyFileOrDir("training_artifacts/optimizer_model.onnx")

        val trainer = ORTTrainer(checkpointPath, trainingModelPath, evalModelPath,
                                 optimizerModelPath)
    }
}
