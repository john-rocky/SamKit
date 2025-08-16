package com.samkit.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.samkit.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Interactive view for SAM segmentation on Android
 */
@Composable
fun SamView(
    bitmap: Bitmap,
    model: SamModelRef,
    config: RuntimeConfig = RuntimeConfig.Best,
    modifier: Modifier = Modifier
) {
    val viewModel: SamViewModel = viewModel {
        SamViewModel(bitmap, model, config)
    }
    
    val uiState by viewModel.uiState.collectAsState()
    
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    
    Box(modifier = modifier.fillMaxSize()) {
        // Image canvas with masks
        ImageCanvas(
            bitmap = bitmap,
            masks = uiState.masks,
            points = uiState.points,
            box = uiState.currentBox,
            selectedMaskIndex = uiState.selectedMaskIndex,
            scale = scale,
            offset = offset,
            onTap = { location ->
                viewModel.addPoint(location)
            },
            onDrag = { start, end ->
                viewModel.setBox(start, end)
            },
            onTransform = { scaleChange, offsetChange ->
                scale *= scaleChange
                offset += offsetChange
            },
            modifier = Modifier.fillMaxSize()
        )
        
        // Loading indicator
        if (uiState.isProcessing) {
            CircularProgressIndicator(
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(48.dp)
            )
        }
        
        // Control panel
        ControlPanel(
            viewModel = viewModel,
            uiState = uiState,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
        )
    }
    
    LaunchedEffect(bitmap) {
        viewModel.setImage(bitmap)
    }
}

@Composable
private fun ImageCanvas(
    bitmap: Bitmap,
    masks: List<SamMask>,
    points: List<SamPoint>,
    box: SamBox?,
    selectedMaskIndex: Int,
    scale: Float,
    offset: Offset,
    onTap: (Offset) -> Unit,
    onDrag: (Offset, Offset) -> Unit,
    onTransform: (Float, Offset) -> Unit,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current
    var dragStart by remember { mutableStateOf<Offset?>(null) }
    
    val transformableState = rememberTransformableState { zoomChange, offsetChange, _ ->
        onTransform(zoomChange, offsetChange)
    }
    
    Canvas(
        modifier = modifier
            .transformable(state = transformableState)
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    val imagePoint = convertToImageCoordinates(
                        offset, scale, this@Canvas.offset, bitmap.width, bitmap.height
                    )
                    onTap(imagePoint)
                }
            }
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = { offset ->
                        dragStart = offset
                    },
                    onDragEnd = {
                        dragStart?.let { start ->
                            val end = start // Last drag position
                            val startImage = convertToImageCoordinates(
                                start, scale, this@Canvas.offset, bitmap.width, bitmap.height
                            )
                            val endImage = convertToImageCoordinates(
                                end, scale, this@Canvas.offset, bitmap.width, bitmap.height
                            )
                            onDrag(startImage, endImage)
                        }
                        dragStart = null
                    }
                ) { _, _ ->
                    // Track drag progress if needed
                }
            }
    ) {
        drawIntoCanvas { canvas ->
            // Apply transformations
            canvas.save()
            canvas.scale(scale)
            canvas.translate(offset.x, offset.y)
            
            // Draw bitmap
            val imageBitmap = bitmap.asImageBitmap()
            canvas.drawImage(
                image = imageBitmap,
                topLeftOffset = Offset.Zero
            )
            
            // Draw masks
            masks.forEachIndexed { index, mask ->
                if (index == selectedMaskIndex || selectedMaskIndex == -1) {
                    drawMask(mask, canvas, 0.5f)
                }
            }
            
            // Draw points
            points.forEach { point ->
                drawPoint(point, canvas)
            }
            
            // Draw box
            box?.let {
                drawBox(it, canvas)
            }
            
            canvas.restore()
        }
    }
}

private fun DrawScope.drawMask(mask: SamMask, canvas: androidx.compose.ui.graphics.Canvas, opacity: Float) {
    val maskBitmap = mask.toColoredBitmap(Color.Blue.toArgb()).asImageBitmap()
    val paint = Paint().apply {
        alpha = opacity
    }
    canvas.drawImage(
        image = maskBitmap,
        topLeftOffset = Offset.Zero,
        paint = paint
    )
}

private fun DrawScope.drawPoint(point: SamPoint, canvas: androidx.compose.ui.graphics.Canvas) {
    val color = if (point.label == 1) Color.Green else Color.Red
    val paint = Paint().apply {
        this.color = color
        style = PaintingStyle.Fill
    }
    
    canvas.drawCircle(
        center = Offset(point.x, point.y),
        radius = 5f,
        paint = paint
    )
}

private fun DrawScope.drawBox(box: SamBox, canvas: androidx.compose.ui.graphics.Canvas) {
    val paint = Paint().apply {
        color = Color.Yellow
        style = PaintingStyle.Stroke
        strokeWidth = 2f
    }
    
    canvas.drawRect(
        offset = Offset(box.x0, box.y0),
        size = Size(box.x1 - box.x0, box.y1 - box.y0),
        paint = paint
    )
}

private fun convertToImageCoordinates(
    viewPoint: Offset,
    scale: Float,
    offset: Offset,
    imageWidth: Int,
    imageHeight: Int
): Offset {
    return Offset(
        x = (viewPoint.x - offset.x) / scale,
        y = (viewPoint.y - offset.y) / scale
    )
}

@Composable
private fun ControlPanel(
    viewModel: SamViewModel,
    uiState: SamUiState,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Mask selection
            if (uiState.masks.size > 1) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    FilterChip(
                        selected = uiState.selectedMaskIndex == -1,
                        onClick = { viewModel.selectMask(-1) },
                        label = { Text("All") }
                    )
                    uiState.masks.forEachIndexed { index, _ ->
                        FilterChip(
                            selected = uiState.selectedMaskIndex == index,
                            onClick = { viewModel.selectMask(index) },
                            label = { Text("Mask ${index + 1}") }
                        )
                    }
                }
            }
            
            // Threshold slider
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Threshold", modifier = Modifier.width(80.dp))
                Slider(
                    value = uiState.maskThreshold,
                    onValueChange = { viewModel.updateThreshold(it) },
                    valueRange = -10f..10f,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    String.format("%.1f", uiState.maskThreshold),
                    modifier = Modifier.width(40.dp)
                )
            }
            
            // Action buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                TextButton(onClick = { viewModel.clearPoints() }) {
                    Text("Clear Points")
                }
                TextButton(onClick = { viewModel.clearBox() }) {
                    Text("Clear Box")
                }
                TextButton(
                    onClick = { viewModel.undo() },
                    enabled = uiState.canUndo
                ) {
                    Text("Undo")
                }
                TextButton(
                    onClick = { viewModel.exportMask() },
                    enabled = uiState.masks.isNotEmpty()
                ) {
                    Text("Export")
                }
            }
        }
    }
}

// View Model
class SamViewModel(
    private val initialBitmap: Bitmap,
    private val model: SamModelRef,
    private val config: RuntimeConfig
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(SamUiState())
    val uiState: StateFlow<SamUiState> = _uiState.asStateFlow()
    
    private var session: SamSession? = null
    private var debounceJob: Job? = null
    private val history = mutableListOf<HistoryEntry>()
    
    init {
        initSession()
    }
    
    private fun initSession() {
        viewModelScope.launch {
            try {
                session = SamSession(model, config)
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
    
    fun setImage(bitmap: Bitmap) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isProcessing = true)
            try {
                session?.setImage(bitmap)
            } catch (e: Exception) {
                // Handle error
            }
            _uiState.value = _uiState.value.copy(isProcessing = false)
        }
    }
    
    fun addPoint(location: Offset) {
        val point = SamPoint(location.x, location.y, 1)
        val newPoints = _uiState.value.points + point
        _uiState.value = _uiState.value.copy(points = newPoints)
        saveHistory()
        debouncedPredict()
    }
    
    fun setBox(start: Offset, end: Offset) {
        val box = SamBox(
            x0 = minOf(start.x, end.x),
            y0 = minOf(start.y, end.y),
            x1 = maxOf(start.x, end.x),
            y1 = maxOf(start.y, end.y)
        )
        _uiState.value = _uiState.value.copy(currentBox = box)
        saveHistory()
        debouncedPredict()
    }
    
    fun clearPoints() {
        _uiState.value = _uiState.value.copy(points = emptyList())
        debouncedPredict()
    }
    
    fun clearBox() {
        _uiState.value = _uiState.value.copy(currentBox = null)
        debouncedPredict()
    }
    
    fun selectMask(index: Int) {
        _uiState.value = _uiState.value.copy(selectedMaskIndex = index)
    }
    
    fun updateThreshold(threshold: Float) {
        _uiState.value = _uiState.value.copy(maskThreshold = threshold)
        debouncedPredict()
    }
    
    fun undo() {
        if (history.size > 1) {
            history.removeLast()
            history.lastOrNull()?.let { entry ->
                _uiState.value = _uiState.value.copy(
                    points = entry.points,
                    currentBox = entry.box
                )
                debouncedPredict()
            }
        }
        updateUndoState()
    }
    
    fun exportMask() {
        val masks = _uiState.value.masks
        val selectedIndex = _uiState.value.selectedMaskIndex
        val mask = if (selectedIndex >= 0 && selectedIndex < masks.size) {
            masks[selectedIndex]
        } else {
            masks.firstOrNull()
        }
        
        mask?.let {
            // Export logic - save to gallery, share, etc.
            val bitmap = it.toBitmap()
            // Save bitmap
        }
    }
    
    private fun saveHistory() {
        history.add(
            HistoryEntry(
                points = _uiState.value.points,
                box = _uiState.value.currentBox
            )
        )
        updateUndoState()
    }
    
    private fun updateUndoState() {
        _uiState.value = _uiState.value.copy(canUndo = history.size > 1)
    }
    
    private fun debouncedPredict() {
        debounceJob?.cancel()
        debounceJob = viewModelScope.launch {
            delay(80) // 80ms debounce
            predict()
        }
    }
    
    private suspend fun predict() {
        val points = _uiState.value.points
        val box = _uiState.value.currentBox
        
        if (points.isEmpty() && box == null) {
            _uiState.value = _uiState.value.copy(masks = emptyList())
            return
        }
        
        _uiState.value = _uiState.value.copy(isProcessing = true)
        
        try {
            val result = session?.predict(
                points = points,
                box = box,
                maskInput = null,
                options = SamOptions(
                    multimaskOutput = true,
                    returnLogits = true,
                    maskThreshold = _uiState.value.maskThreshold
                )
            )
            
            result?.let {
                _uiState.value = _uiState.value.copy(masks = it.masks)
            }
        } catch (e: Exception) {
            // Handle error
        }
        
        _uiState.value = _uiState.value.copy(isProcessing = false)
    }
    
    override fun onCleared() {
        super.onCleared()
        session?.close()
    }
}

data class SamUiState(
    val masks: List<SamMask> = emptyList(),
    val points: List<SamPoint> = emptyList(),
    val currentBox: SamBox? = null,
    val selectedMaskIndex: Int = -1,
    val maskThreshold: Float = 0f,
    val isProcessing: Boolean = false,
    val canUndo: Boolean = false
)

private data class HistoryEntry(
    val points: List<SamPoint>,
    val box: SamBox?
)