#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <cstring>
#include <vector>
#include <memory>

#include "samkit/preprocessor.h"
#include "samkit/postprocessor.h"
#include "samkit/types.h"
#include "samkit/image.h"

#define LOG_TAG "SAMKit"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

// Native preprocessing implementation
JNIEXPORT jlong JNICALL
Java_com_samkit_NativePreprocessor_createNative(JNIEnv* env, jobject /* this */, jint modelSize) {
    auto* preprocessor = new samkit::Preprocessor(modelSize);
    return reinterpret_cast<jlong>(preprocessor);
}

JNIEXPORT void JNICALL
Java_com_samkit_NativePreprocessor_destroyNative(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle != 0) {
        auto* preprocessor = reinterpret_cast<samkit::Preprocessor*>(handle);
        delete preprocessor;
    }
}

JNIEXPORT jobject JNICALL
Java_com_samkit_NativePreprocessor_processBitmap(
    JNIEnv* env, 
    jobject /* this */, 
    jlong handle,
    jobject bitmap
) {
    if (handle == 0 || bitmap == nullptr) {
        return nullptr;
    }
    
    auto* preprocessor = reinterpret_cast<samkit::Preprocessor*>(handle);
    
    // Get bitmap info
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get bitmap info");
        return nullptr;
    }
    
    // Lock bitmap pixels
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock bitmap pixels");
        return nullptr;
    }
    
    // Convert to SAMKit Image
    samkit::Image image;
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        image = samkit::Image::fromRGBA(
            static_cast<const uint8_t*>(pixels),
            info.width,
            info.height
        );
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        // Convert RGB565 to RGB888
        std::vector<uint8_t> rgb888(info.width * info.height * 3);
        const uint16_t* src = static_cast<const uint16_t*>(pixels);
        
        for (size_t i = 0; i < info.width * info.height; ++i) {
            uint16_t pixel = src[i];
            rgb888[i * 3 + 0] = ((pixel >> 11) & 0x1F) << 3; // R
            rgb888[i * 3 + 1] = ((pixel >> 5) & 0x3F) << 2;  // G
            rgb888[i * 3 + 2] = (pixel & 0x1F) << 3;         // B
        }
        
        image = samkit::Image::fromRGB(rgb888.data(), info.width, info.height);
    }
    
    // Unlock bitmap
    AndroidBitmap_unlockPixels(env, bitmap);
    
    // Process image
    auto [tensor, transform] = preprocessor->process(image);
    
    // Convert tensor to Java ByteBuffer
    jclass byteBufferClass = env->FindClass("java/nio/ByteBuffer");
    jmethodID allocateDirect = env->GetStaticMethodID(
        byteBufferClass, 
        "allocateDirect", 
        "(I)Ljava/nio/ByteBuffer;"
    );
    
    size_t bufferSize = tensor.bytes();
    jobject buffer = env->CallStaticObjectMethod(
        byteBufferClass,
        allocateDirect,
        static_cast<jint>(bufferSize)
    );
    
    // Copy tensor data to buffer
    void* bufferData = env->GetDirectBufferAddress(buffer);
    if (bufferData != nullptr) {
        tensor.copyTo(bufferData);
    }
    
    // Create TransformParams object
    jclass transformClass = env->FindClass("com/samkit/TransformParams");
    jmethodID transformConstructor = env->GetMethodID(
        transformClass,
        "<init>",
        "(FFFIII)V"
    );
    
    jobject transformObj = env->NewObject(
        transformClass,
        transformConstructor,
        transform.scale,
        transform.pad_x,
        transform.pad_y,
        transform.original_width,
        transform.original_height,
        transform.model_size
    );
    
    // Create Pair object
    jclass pairClass = env->FindClass("kotlin/Pair");
    jmethodID pairConstructor = env->GetMethodID(
        pairClass,
        "<init>",
        "(Ljava/lang/Object;Ljava/lang/Object;)V"
    );
    
    jobject pair = env->NewObject(
        pairClass,
        pairConstructor,
        buffer,
        transformObj
    );
    
    return pair;
}

// Native postprocessing implementation
JNIEXPORT jlong JNICALL
Java_com_samkit_NativePostprocessor_createNative(JNIEnv* env, jobject /* this */, jint modelSize) {
    auto* postprocessor = new samkit::Postprocessor(modelSize);
    return reinterpret_cast<jlong>(postprocessor);
}

JNIEXPORT void JNICALL
Java_com_samkit_NativePostprocessor_destroyNative(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle != 0) {
        auto* postprocessor = reinterpret_cast<samkit::Postprocessor*>(handle);
        delete postprocessor;
    }
}

JNIEXPORT jobject JNICALL
Java_com_samkit_NativePostprocessor_processMasks(
    JNIEnv* env,
    jobject /* this */,
    jlong handle,
    jobject maskBuffer,
    jobject scoreBuffer,
    jobject transformObj,
    jobject optionsObj
) {
    if (handle == 0) {
        return nullptr;
    }
    
    auto* postprocessor = reinterpret_cast<samkit::Postprocessor*>(handle);
    
    // Get buffer data
    float* maskData = static_cast<float*>(env->GetDirectBufferAddress(maskBuffer));
    float* scoreData = static_cast<float*>(env->GetDirectBufferAddress(scoreBuffer));
    jlong maskSize = env->GetDirectBufferCapacity(maskBuffer);
    jlong scoreSize = env->GetDirectBufferCapacity(scoreBuffer);
    
    if (maskData == nullptr || scoreData == nullptr) {
        LOGE("Failed to get buffer addresses");
        return nullptr;
    }
    
    // Convert to tensors
    int numMasks = scoreSize / sizeof(float);
    int maskHeight = 256; // Typical for MobileSAM
    int maskWidth = 256;
    
    samkit::Tensor maskTensor = samkit::Tensor::fromFloat(
        maskData,
        {numMasks, 1, maskHeight, maskWidth}
    );
    
    samkit::Tensor scoreTensor = samkit::Tensor::fromFloat(
        scoreData,
        {numMasks}
    );
    
    // Get transform params
    jclass transformClass = env->GetObjectClass(transformObj);
    jfieldID scaleField = env->GetFieldID(transformClass, "scale", "F");
    jfieldID padXField = env->GetFieldID(transformClass, "padX", "F");
    jfieldID padYField = env->GetFieldID(transformClass, "padY", "F");
    jfieldID origWidthField = env->GetFieldID(transformClass, "originalWidth", "I");
    jfieldID origHeightField = env->GetFieldID(transformClass, "originalHeight", "I");
    jfieldID modelSizeField = env->GetFieldID(transformClass, "modelSize", "I");
    
    samkit::TransformParams transform;
    transform.scale = env->GetFloatField(transformObj, scaleField);
    transform.pad_x = env->GetFloatField(transformObj, padXField);
    transform.pad_y = env->GetFloatField(transformObj, padYField);
    transform.original_width = env->GetIntField(transformObj, origWidthField);
    transform.original_height = env->GetIntField(transformObj, origHeightField);
    transform.model_size = env->GetIntField(transformObj, modelSizeField);
    
    // Get options
    jclass optionsClass = env->GetObjectClass(optionsObj);
    jfieldID multimaskField = env->GetFieldID(optionsClass, "multimaskOutput", "Z");
    jfieldID returnLogitsField = env->GetFieldID(optionsClass, "returnLogits", "Z");
    jfieldID thresholdField = env->GetFieldID(optionsClass, "maskThreshold", "F");
    jfieldID maxMasksField = env->GetFieldID(optionsClass, "maxMasks", "I");
    
    samkit::Options options;
    options.multimask_output = env->GetBooleanField(optionsObj, multimaskField);
    options.return_logits = env->GetBooleanField(optionsObj, returnLogitsField);
    options.mask_threshold = env->GetFloatField(optionsObj, thresholdField);
    options.max_masks = env->GetIntField(optionsObj, maxMasksField);
    
    // Process masks
    samkit::Result result = postprocessor->process(
        maskTensor,
        scoreTensor,
        transform,
        options
    );
    
    // Convert result to Java object
    jclass resultClass = env->FindClass("com/samkit/SamResult");
    jclass maskClass = env->FindClass("com/samkit/SamMask");
    jclass listClass = env->FindClass("java/util/ArrayList");
    
    jmethodID resultConstructor = env->GetMethodID(
        resultClass,
        "<init>",
        "(Ljava/util/List;ZLjava/lang/Throwable;)V"
    );
    
    jmethodID maskConstructor = env->GetMethodID(
        maskClass,
        "<init>",
        "(II[F[BF)V"
    );
    
    jmethodID listConstructor = env->GetMethodID(listClass, "<init>", "()V");
    jmethodID listAdd = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
    
    // Create mask list
    jobject maskList = env->NewObject(listClass, listConstructor);
    
    for (const auto& mask : result.masks) {
        // Convert logits to Java array if present
        jfloatArray logitsArray = nullptr;
        if (!mask.logits.empty()) {
            logitsArray = env->NewFloatArray(mask.logits.size());
            env->SetFloatArrayRegion(
                logitsArray,
                0,
                mask.logits.size(),
                mask.logits.data()
            );
        }
        
        // Convert alpha to Java byte array
        jbyteArray alphaArray = env->NewByteArray(mask.alpha.size());
        env->SetByteArrayRegion(
            alphaArray,
            0,
            mask.alpha.size(),
            reinterpret_cast<const jbyte*>(mask.alpha.data())
        );
        
        // Create mask object
        jobject maskObj = env->NewObject(
            maskClass,
            maskConstructor,
            mask.width,
            mask.height,
            logitsArray,
            alphaArray,
            mask.score
        );
        
        env->CallBooleanMethod(maskList, listAdd, maskObj);
        
        // Clean up local references
        if (logitsArray != nullptr) {
            env->DeleteLocalRef(logitsArray);
        }
        env->DeleteLocalRef(alphaArray);
        env->DeleteLocalRef(maskObj);
    }
    
    // Create result object
    jobject resultObj = env->NewObject(
        resultClass,
        resultConstructor,
        maskList,
        result.success,
        nullptr // No error for successful result
    );
    
    return resultObj;
}

} // extern "C"