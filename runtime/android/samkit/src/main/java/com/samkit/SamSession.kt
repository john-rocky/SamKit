package com.samkit

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Main session class for SAM inference on Android
 */
class SamSession(
    private val model: SamModelRef,
    private val config: RuntimeConfig = RuntimeConfig.Best
) {
    private var encoder: Interpreter? = null
    private var decoder: Interpreter? = null
    private var cachedEmbedding: FloatBuffer? = null
    private var transformParams: TransformParams? = null
    private val modelSize = model.inputSize
    private val preprocessor = Preprocessor(modelSize)
    private val postprocessor = Postprocessor(modelSize)
    
    // Delegates for hardware acceleration
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    
    init {
        initializeInterpreters()
    }
    
    private fun initializeInterpreters() {
        val options = Interpreter.Options().apply {
            setNumThreads(config.numThreads)
            
            // Add hardware acceleration if available
            when (config.computeUnit) {
                RuntimeConfig.ComputeUnit.GPU_PREFERRED -> {
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                    } catch (e: Exception) {
                        // Fallback to CPU if GPU not available
                    }
                }
                RuntimeConfig.ComputeUnit.NEURAL_ENGINE_PREFERRED -> {
                    try {
                        nnApiDelegate = NnApiDelegate()
                        addDelegate(nnApiDelegate)
                    } catch (e: Exception) {
                        // Fallback to CPU if NNAPI not available
                    }
                }
                RuntimeConfig.ComputeUnit.BEST_AVAILABLE -> {
                    // Try GPU first, then NNAPI, then CPU
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate)
                    } catch (e: Exception) {
                        try {
                            nnApiDelegate = NnApiDelegate()
                            addDelegate(nnApiDelegate)
                        } catch (e2: Exception) {
                            // Use CPU
                        }
                    }
                }
                else -> {
                    // CPU only
                }
            }
        }
        
        // Load models
        encoder = Interpreter(model.encoderBuffer, options)
        decoder = Interpreter(model.decoderBuffer, options)
    }
    
    /**
     * Set the image for segmentation
     */
    suspend fun setImage(bitmap: Bitmap) = withContext(Dispatchers.Default) {
        // Clear previous cache
        cachedEmbedding = null
        transformParams = null
        
        // Preprocess image
        val (inputBuffer, transform) = preprocessor.process(bitmap)
        transformParams = transform
        
        // Prepare encoder input/output buffers
        val embeddingSize = 64 // Typical for MobileSAM
        val embeddingChannels = 256
        val outputBuffer = ByteBuffer.allocateDirect(
            4 * embeddingChannels * embeddingSize * embeddingSize
        ).order(ByteOrder.nativeOrder())
        
        // Run encoder
        encoder?.run(inputBuffer, outputBuffer)
        
        // Cache embedding
        cachedEmbedding = outputBuffer.asFloatBuffer()
    }
    
    /**
     * Run mask prediction with prompts
     */
    suspend fun predict(
        points: List<SamPoint> = emptyList(),
        box: SamBox? = null,
        maskInput: SamMaskRef? = null,
        options: SamOptions = SamOptions()
    ): SamResult = withContext(Dispatchers.Default) {
        
        val embedding = cachedEmbedding ?: throw IllegalStateException("Image not set. Call setImage() first.")
        val transform = transformParams ?: throw IllegalStateException("Transform params not available")
        
        // Encode prompts
        val (pointCoords, pointLabels) = preprocessor.encodePoints(points, transform)
        
        // Prepare decoder inputs
        val inputs = mutableMapOf<String, Any>()
        inputs["image_embeddings"] = embedding.duplicate()
        inputs["point_coords"] = pointCoords
        inputs["point_labels"] = pointLabels
        
        // Prepare outputs
        val numMasks = 3 // MobileSAM typically outputs 3 masks
        val maskSize = 256
        val masksBuffer = ByteBuffer.allocateDirect(
            4 * numMasks * maskSize * maskSize
        ).order(ByteOrder.nativeOrder())
        
        val scoresBuffer = ByteBuffer.allocateDirect(
            4 * numMasks
        ).order(ByteOrder.nativeOrder())
        
        val outputs = mapOf(
            0 to masksBuffer,
            1 to scoresBuffer
        )
        
        // Run decoder
        decoder?.runForMultipleInputsOutputs(
            arrayOf(embedding.duplicate(), pointCoords, pointLabels),
            outputs
        )
        
        // Postprocess results
        postprocessor.process(
            masksBuffer.asFloatBuffer(),
            scoresBuffer.asFloatBuffer(),
            transform,
            options
        )
    }
    
    /**
     * Clear cached embedding
     */
    fun clear() {
        cachedEmbedding = null
        transformParams = null
    }
    
    /**
     * Release resources
     */
    fun close() {
        encoder?.close()
        decoder?.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
        
        encoder = null
        decoder = null
        gpuDelegate = null
        nnApiDelegate = null
    }
}

/**
 * Model reference for loading TFLite models
 */
data class SamModelRef(
    val encoderBuffer: ByteBuffer,
    val decoderBuffer: ByteBuffer,
    val inputSize: Int = 1024,
    val modelType: ModelType = ModelType.MOBILE_SAM
) {
    companion object {
        /**
         * Load model from assets
         */
        fun fromAssets(
            assets: android.content.res.AssetManager,
            modelType: ModelType = ModelType.MOBILE_SAM
        ): SamModelRef {
            val encoderPath = "${modelType.modelName}_encoder.tflite"
            val decoderPath = "${modelType.modelName}_decoder.tflite"
            
            val encoderBuffer = loadModelFile(assets, encoderPath)
            val decoderBuffer = loadModelFile(assets, decoderPath)
            
            return SamModelRef(
                encoderBuffer = encoderBuffer,
                decoderBuffer = decoderBuffer,
                modelType = modelType
            )
        }
        
        private fun loadModelFile(assets: android.content.res.AssetManager, path: String): ByteBuffer {
            assets.openFd(path).use { fileDescriptor ->
                val inputStream = fileDescriptor.createInputStream()
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                return fileChannel.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY,
                    startOffset,
                    declaredLength
                )
            }
        }
    }
}

enum class ModelType(val modelName: String) {
    MOBILE_SAM("mobile_sam")
}

/**
 * Runtime configuration
 */
data class RuntimeConfig(
    val computeUnit: ComputeUnit = ComputeUnit.BEST_AVAILABLE,
    val numThreads: Int = 4,
    val enableFP16: Boolean = true,
    val enableQuantization: Boolean = false
) {
    enum class ComputeUnit {
        CPU_ONLY,
        GPU_PREFERRED,
        NEURAL_ENGINE_PREFERRED,
        BEST_AVAILABLE
    }
    
    companion object {
        val Best = RuntimeConfig()
    }
}

/**
 * Point prompt for SAM
 */
data class SamPoint(
    val x: Float,
    val y: Float,
    val label: Int = 1  // 1 = positive, 0 = negative
)

/**
 * Bounding box
 */
data class SamBox(
    val x0: Float,
    val y0: Float,
    val x1: Float,
    val y1: Float
)

/**
 * SAM inference options
 */
data class SamOptions(
    val multimaskOutput: Boolean = true,
    val returnLogits: Boolean = false,
    val maskThreshold: Float = 0.0f,
    val maxMasks: Int = 3
)

/**
 * Single mask result
 */
data class SamMask(
    val width: Int,
    val height: Int,
    val logits: FloatArray? = null,
    val alpha: ByteArray,
    val score: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SamMask

        if (width != other.width) return false
        if (height != other.height) return false
        if (logits != null) {
            if (other.logits == null) return false
            if (!logits.contentEquals(other.logits)) return false
        } else if (other.logits != null) return false
        if (!alpha.contentEquals(other.alpha)) return false
        if (score != other.score) return false

        return true
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + (logits?.contentHashCode() ?: 0)
        result = 31 * result + alpha.contentHashCode()
        result = 31 * result + score.hashCode()
        return result
    }
}

typealias SamMaskRef = SamMask

/**
 * SAM inference result
 */
data class SamResult(
    val masks: List<SamMask>,
    val success: Boolean = true,
    val error: Throwable? = null
)

/**
 * Transform parameters for coordinate mapping
 */
data class TransformParams(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val originalWidth: Int,
    val originalHeight: Int,
    val modelSize: Int
) {
    fun toModel(point: SamPoint): SamPoint {
        return SamPoint(
            x = point.x * scale + padX,
            y = point.y * scale + padY,
            label = point.label
        )
    }
    
    fun toImage(point: SamPoint): SamPoint {
        return SamPoint(
            x = (point.x - padX) / scale,
            y = (point.y - padY) / scale,
            label = point.label
        )
    }
}