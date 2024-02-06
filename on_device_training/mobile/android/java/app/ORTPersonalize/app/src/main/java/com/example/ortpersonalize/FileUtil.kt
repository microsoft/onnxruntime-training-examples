package com.example.ortpersonalize

import android.content.res.AssetManager
import java.io.File

fun copyAssetFile(assetManager: AssetManager, assetPath: String, dstFile: File) {
    // This function copies the asset file named by `assetPath` to the file specified by `dstFile`.
    check(!dstFile.exists() || dstFile.isFile)

    dstFile.parentFile?.mkdirs()

    val assetContents = assetManager.open(assetPath).use { assetStream ->
        val size: Int = assetStream.available()
        val buffer = ByteArray(size)
        assetStream.read(buffer)
        buffer
    }

    java.io.FileOutputStream(dstFile).use { dstStream ->
        dstStream.write(assetContents)
    }
}

fun copyAssetFileOrDir(assetManager: AssetManager, assetPath: String, dstFileOrDir: File) {
    // This function copies the asset file or directory named by `assetPath` to the file or
    // directory specified by `dstFileOrDir`.
    val assets: Array<String>? = assetManager.list(assetPath)
    if (assets!!.isEmpty()) {
        // asset is a file
        copyAssetFile(assetManager, assetPath, dstFileOrDir)
    } else {
        // asset is a dir. loop over dir and copy all files or sub dirs to cache dir
        for (i in assets.indices) {
            val assetChild = (if (assetPath.isEmpty()) "" else "$assetPath/") + assets[i]
            val dstChild = dstFileOrDir.resolve(assets[i])
            copyAssetFileOrDir(assetManager, assetChild, dstChild)
        }
    }
}
