//
// Created by bmeswani on 2/16/2023.
//

#include "utils.h"

namespace utils {

    std::string JString2String(JNIEnv *env, jstring jStr) {
        if (!jStr)
            return std::string();

        const jclass stringClass = env->GetObjectClass(jStr);
        const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes",
                                                    "(Ljava/lang/String;)[B");
        const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                           env->NewStringUTF(
                                                                                   "UTF-8"));

        size_t length = (size_t) env->GetArrayLength(stringJbytes);
        jbyte *pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

        std::string ret = std::string((char *) pBytes, length);
        env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

        env->DeleteLocalRef(stringJbytes);
        env->DeleteLocalRef(stringClass);
        return ret;
    }

} // utils
