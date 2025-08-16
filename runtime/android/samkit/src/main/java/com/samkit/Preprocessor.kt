package com.samkit

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max

/**
 * Handles image preprocessing for SAM models
 */
internal class Preprocessor(private val modelSize: Int = 1024) {
    
    private val mean = floatArrayOf(123.675f, 116.28f, 103.53f)
    private val std = floatArrayOf(58.395f, 57.12f, 57.375f)
    
    /**
     * Process bitmap for model input
     */
    fun process(bitmap: Bitmap): Pair<ByteBuffer, TransformParams> {
        // Calculate transform parameters
        val transform = computeTransform(bitmap.width, bitmap.height)
        
        // Resize and pad bitmap
        val resizedBitmap = resizeAndPad(bitmap, transform)
        
        // Convert to ByteBuffer
        val buffer = bitmapToByteBuffer(resizedBitmap)
        
        // Normalize
        normalize(buffer)
        
        return Pair(buffer, transform)
    }
    
    /**
     * Encode point prompts
     */
    fun encodePoints(
        points: List<SamPoint>,
        transform: TransformParams
    ): Pair<FloatBuffer, FloatBuffer> {
        val count = max(1, points.size)
        
        // Create coordinate buffer (N, 2)
        val coordsBuffer = ByteBuffer.allocateDirect(4 * count * 2)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        
        // Create label buffer (N,)
        val labelsBuffer = ByteBuffer.allocateDirect(4 * count)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        
        if (points.isEmpty()) {
            // Fill with dummy point at center
            coordsBuffer.put(modelSize / 2f)
            coordsBuffer.put(modelSize / 2f)
            labelsBuffer.put(-1f) // Invalid label
        } else {
            for (point in points) {
                // Transform to model coordinates
                val modelPoint = transform.toModel(point)
                coordsBuffer.put(modelPoint.x)
                coordsBuffer.put(modelPoint.y)
                labelsBuffer.put(point.label.toFloat())
            }
        }
        
        coordsBuffer.rewind()
        labelsBuffer.rewind()
        
        return Pair(coordsBuffer, labelsBuffer)
    }
    
    /**
     * Encode mask input
     */
    fun encodeMask(mask: SamMaskRef, transform: TransformParams): FloatBuffer {
        val maskSize = 256
        val buffer = ByteBuffer.allocateDirect(4 * maskSize * maskSize)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        
        // Resample mask to 256x256
        val scaleX = mask.width.toFloat() / maskSize
        val scaleY = mask.height.toFloat() / maskSize
        
        for (y in 0 until maskSize) {
            for (x in 0 until maskSize) {
                val srcX = (x * scaleX).toInt()
                val srcY = (y * scaleY).toInt()
                val srcIdx = srcY * mask.width + srcX
                
                val value = when {
                    mask.logits != null && srcIdx < mask.logits.size -> {
                        mask.logits[srcIdx]
                    }
                    srcIdx < mask.alpha.size -> {
                        mask.alpha[srcIdx].toFloat() / 255f
                    }
                    else -> 0f
                }
                
                buffer.put(value)
            }
        }
        
        buffer.rewind()
        return buffer
    }
    
    private fun computeTransform(originalWidth: Int, originalHeight: Int): TransformParams {
        val longSide = max(originalWidth, originalHeight)
        val scale = modelSize.toFloat() / longSide
        
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()
        
        val padX = (modelSize - scaledWidth) / 2f
        val padY = (modelSize - scaledHeight) / 2f
        
        return TransformParams(
            scale = scale,
            padX = padX,
            padY = padY,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            modelSize = modelSize
        )
    }
    
    private fun resizeAndPad(bitmap: Bitmap, transform: TransformParams): Bitmap {
        val scaledWidth = (bitmap.width * transform.scale).toInt()
        val scaledHeight = (bitmap.height * transform.scale).toInt()
        
        // Create padded bitmap
        val paddedBitmap = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        
        // Fill with black (padding)
        canvas.drawColor(Color.BLACK)
        
        // Draw scaled bitmap at center
        val paint = Paint().apply {
            isFilterBitmap = true
            isAntiAlias = true
        }
        
        val src = Rect(0, 0, bitmap.width, bitmap.height)
        val dst = Rect(
            transform.padX.toInt(),
            transform.padY.toInt(),
            transform.padX.toInt() + scaledWidth,
            transform.padY.toInt() + scaledHeight
        )
        
        canvas.drawBitmap(bitmap, src, dst, paint)
        
        return paddedBitmap
    }
    
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * 3 * modelSize * modelSize)
            .order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(modelSize * modelSize)
        bitmap.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
        
        // Convert to CHW format
        val channelSize = modelSize * modelSize
        
        for (y in 0 until modelSize) {
            for (x in 0 until modelSize) {
                val pixel = pixels[y * modelSize + x]
                
                // Extract RGB values
                val r = ((pixel shr 16) and 0xFF).toFloat()
                val g = ((pixel shr 8) and 0xFF).toFloat()
                val b = (pixel and 0xFF).toFloat()
                
                // Write in CHW format
                buffer.putFloat(r)
            }
        }
        
        for (y in 0 until modelSize) {
            for (x in 0 until modelSize) {
                val pixel = pixels[y * modelSize + x]
                val g = ((pixel shr 8) and 0xFF).toFloat()
                buffer.putFloat(g)
            }
        }
        
        for (y in 0 until modelSize) {
            for (x in 0 until modelSize) {
                val pixel = pixels[y * modelSize + x]
                val b = (pixel and 0xFF).toFloat()
                buffer.putFloat(b)
            }
        }
        
        buffer.rewind()
        return buffer
    }
    
    private fun normalize(buffer: ByteBuffer) {
        val floatBuffer = buffer.asFloatBuffer()
        val channelSize = modelSize * modelSize
        
        // Normalize each channel
        for (c in 0 until 3) {
            val channelStart = c * channelSize
            for (i in 0 until channelSize) {
                val idx = channelStart + i
                val value = floatBuffer.get(idx)
                floatBuffer.put(idx, (value - mean[c]) / std[c])
            }
        }
    }
}