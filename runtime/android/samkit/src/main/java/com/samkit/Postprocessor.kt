package com.samkit

import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Handles mask postprocessing for SAM models
 */
internal class Postprocessor(private val modelSize: Int = 1024) {
    
    /**
     * Process model output into final masks
     */
    fun process(
        maskLogits: FloatBuffer,
        iouPredictions: FloatBuffer,
        transform: TransformParams,
        options: SamOptions
    ): SamResult {
        
        try {
            // 1. Upsample masks from low resolution to model size
            val upsampled = upsampleMasks(maskLogits)
            
            // 2. Remove padding
            val depadded = removePadding(upsampled, transform)
            
            // 3. Resize to original image size
            val finalMasks = resizeToOriginal(depadded, transform)
            
            // 4. Extract individual masks with scores
            val masks = extractMasks(finalMasks, iouPredictions, options)
            
            return SamResult(masks = masks)
        } catch (e: Exception) {
            return SamResult(
                masks = emptyList(),
                success = false,
                error = e
            )
        }
    }
    
    private fun upsampleMasks(masks: FloatBuffer): FloatArray {
        // Assuming masks shape is (3, 256, 256) for MobileSAM
        val numMasks = 3
        val lowRes = 256
        val scaleFactor = modelSize / lowRes
        
        val upsampled = FloatArray(numMasks * modelSize * modelSize)
        
        for (n in 0 until numMasks) {
            val maskOffset = n * lowRes * lowRes
            val upsampledOffset = n * modelSize * modelSize
            
            // Bilinear upsampling
            for (y in 0 until modelSize) {
                val srcY = y.toFloat() / scaleFactor
                val y0 = srcY.toInt()
                val y1 = min(y0 + 1, lowRes - 1)
                val dy = srcY - y0
                
                for (x in 0 until modelSize) {
                    val srcX = x.toFloat() / scaleFactor
                    val x0 = srcX.toInt()
                    val x1 = min(x0 + 1, lowRes - 1)
                    val dx = srcX - x0
                    
                    // Get values from input mask
                    val v00 = masks.get(maskOffset + y0 * lowRes + x0)
                    val v01 = masks.get(maskOffset + y0 * lowRes + x1)
                    val v10 = masks.get(maskOffset + y1 * lowRes + x0)
                    val v11 = masks.get(maskOffset + y1 * lowRes + x1)
                    
                    // Bilinear interpolation
                    val v0 = v00 * (1 - dx) + v01 * dx
                    val v1 = v10 * (1 - dx) + v11 * dx
                    val v = v0 * (1 - dy) + v1 * dy
                    
                    upsampled[upsampledOffset + y * modelSize + x] = v
                }
            }
        }
        
        return upsampled
    }
    
    private fun removePadding(masks: FloatArray, transform: TransformParams): FloatArray {
        val numMasks = masks.size / (modelSize * modelSize)
        
        // Calculate the actual image size in model space
        val scaledWidth = (transform.originalWidth * transform.scale).toInt()
        val scaledHeight = (transform.originalHeight * transform.scale).toInt()
        
        // Calculate padding
        val padLeft = transform.padX.toInt()
        val padTop = transform.padY.toInt()
        
        val depadded = FloatArray(numMasks * scaledHeight * scaledWidth)
        
        // Copy unpadded region for each mask
        for (n in 0 until numMasks) {
            val maskOffset = n * modelSize * modelSize
            val depaddedOffset = n * scaledHeight * scaledWidth
            
            for (y in 0 until scaledHeight) {
                for (x in 0 until scaledWidth) {
                    val srcIdx = maskOffset + (y + padTop) * modelSize + (x + padLeft)
                    val dstIdx = depaddedOffset + y * scaledWidth + x
                    depadded[dstIdx] = masks[srcIdx]
                }
            }
        }
        
        return depadded
    }
    
    private fun resizeToOriginal(masks: FloatArray, transform: TransformParams): FloatArray {
        val scaledWidth = (transform.originalWidth * transform.scale).toInt()
        val scaledHeight = (transform.originalHeight * transform.scale).toInt()
        val numMasks = masks.size / (scaledWidth * scaledHeight)
        
        val finalMasks = FloatArray(numMasks * transform.originalHeight * transform.originalWidth)
        
        // Resize each mask to original size using bilinear interpolation
        val xScale = scaledWidth.toFloat() / transform.originalWidth
        val yScale = scaledHeight.toFloat() / transform.originalHeight
        
        for (n in 0 until numMasks) {
            val maskOffset = n * scaledHeight * scaledWidth
            val finalOffset = n * transform.originalHeight * transform.originalWidth
            
            for (y in 0 until transform.originalHeight) {
                val srcY = y * yScale
                val y0 = srcY.toInt()
                val y1 = min(y0 + 1, scaledHeight - 1)
                val dy = srcY - y0
                
                for (x in 0 until transform.originalWidth) {
                    val srcX = x * xScale
                    val x0 = srcX.toInt()
                    val x1 = min(x0 + 1, scaledWidth - 1)
                    val dx = srcX - x0
                    
                    // Bilinear interpolation
                    val v00 = masks[maskOffset + y0 * scaledWidth + x0]
                    val v01 = masks[maskOffset + y0 * scaledWidth + x1]
                    val v10 = masks[maskOffset + y1 * scaledWidth + x0]
                    val v11 = masks[maskOffset + y1 * scaledWidth + x1]
                    
                    val v0 = v00 * (1 - dx) + v01 * dx
                    val v1 = v10 * (1 - dx) + v11 * dx
                    val v = v0 * (1 - dy) + v1 * dy
                    
                    finalMasks[finalOffset + y * transform.originalWidth + x] = v
                }
            }
        }
        
        return finalMasks
    }
    
    private fun extractMasks(
        masks: FloatArray,
        scores: FloatBuffer,
        options: SamOptions
    ): List<SamMask> {
        
        val numMasks = scores.remaining()
        val height = masks.size / (numMasks * masks.size / numMasks)
        val width = masks.size / (numMasks * height)
        
        // Get scores and sort by them
        val scoreArray = FloatArray(numMasks)
        for (i in 0 until numMasks) {
            scoreArray[i] = scores.get(i)
        }
        val sortedIndices = scoreArray.indices.sortedByDescending { scoreArray[it] }
        
        // Determine how many masks to return
        val masksToReturn = if (options.multimaskOutput) {
            min(options.maxMasks, numMasks)
        } else {
            1
        }
        
        val resultMasks = mutableListOf<SamMask>()
        
        for (i in 0 until masksToReturn) {
            val idx = sortedIndices[i]
            val maskOffset = idx * height * width
            
            // Prepare logits if requested
            val logits = if (options.returnLogits) {
                FloatArray(height * width) { j ->
                    masks[maskOffset + j]
                }
            } else {
                null
            }
            
            // Convert to alpha mask
            val alpha = createAlphaMask(
                masks,
                maskOffset,
                width * height,
                options.maskThreshold
            )
            
            val mask = SamMask(
                width = width,
                height = height,
                logits = logits,
                alpha = alpha,
                score = scoreArray[idx]
            )
            
            resultMasks.add(mask)
        }
        
        return resultMasks
    }
    
    private fun createAlphaMask(
        logits: FloatArray,
        offset: Int,
        size: Int,
        threshold: Float
    ): ByteArray {
        val alpha = ByteArray(size)
        
        if (threshold > 0) {
            // Apply threshold
            for (i in 0 until size) {
                alpha[i] = if (logits[offset + i] > threshold) {
                    255.toByte()
                } else {
                    0
                }
            }
        } else {
            // Convert logits to alpha using sigmoid
            for (i in 0 until size) {
                val sigmoid = 1.0f / (1.0f + exp(-logits[offset + i]))
                alpha[i] = (sigmoid * 255f).toInt().coerceIn(0, 255).toByte()
            }
        }
        
        return alpha
    }
}

// Extension functions for mask visualization
fun SamMask.toBitmap(): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(width * height)
    
    for (i in alpha.indices) {
        val alphaValue = alpha[i].toInt() and 0xFF
        // Create white pixel with alpha
        pixels[i] = (alphaValue shl 24) or 0xFFFFFF
    }
    
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
}

fun SamMask.toColoredBitmap(color: Int): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(width * height)
    
    val r = (color shr 16) and 0xFF
    val g = (color shr 8) and 0xFF
    val b = color and 0xFF
    
    for (i in alpha.indices) {
        val alphaValue = alpha[i].toInt() and 0xFF
        // Create colored pixel with alpha
        pixels[i] = (alphaValue shl 24) or (r shl 16) or (g shl 8) or b
    }
    
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
}