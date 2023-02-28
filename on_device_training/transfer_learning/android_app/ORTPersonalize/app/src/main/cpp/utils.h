//
// Created by bmeswani on 2/16/2023.
//

#ifndef ORT_PERSONALIZE_UTILS_H
#define ORT_PERSONALIZE_UTILS_H

#include <string>
#include <jni.h>

namespace utils {

    std::string JString2String(JNIEnv *env, jstring jStr);

} // utils

#endif //ORT_PERSONALIZE_UTILS_H
