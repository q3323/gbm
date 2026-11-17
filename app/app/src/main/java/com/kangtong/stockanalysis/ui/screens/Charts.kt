package com.kangtong.stockanalysis.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.kangtong.stockanalysis.data.DailyBar
import com.kangtong.stockanalysis.data.IndicatorSnapshot
import kotlin.math.max
import kotlin.math.min

@Composable
fun CandlestickChart(
    bars: List<DailyBar>,
    modifier: Modifier = Modifier
) {
    val upColor = Color(0xFFE5484D)
    val downColor = Color(0xFF30A46C)
    val gridColor = MaterialTheme.colorScheme.outlineVariant
    val visible = bars.takeLast(60)

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(220.dp)
    ) {
        if (visible.isEmpty()) return@Canvas
        val lows = visible.mapNotNull { it.low }
        val highs = visible.mapNotNull { it.high }
        if (lows.isEmpty() || highs.isEmpty()) return@Canvas

        val minPrice = min(lows.min(), highs.min())
        val maxPrice = max(highs.max(), lows.max())
        val range = (maxPrice - minPrice).takeIf { it > 0.0 } ?: 1.0
        val chartHeight = size.height
        val slotWidth = size.width / visible.size
        val bodyWidth = max(2f, slotWidth * 0.58f)

        for (index in 1..3) {
            val y = chartHeight * index / 4f
            drawLine(
                color = gridColor,
                start = Offset(0f, y),
                end = Offset(size.width, y),
                strokeWidth = 1f
            )
        }

        visible.forEachIndexed { index, bar ->
            val open = bar.open ?: return@forEachIndexed
            val close = bar.close ?: return@forEachIndexed
            val high = bar.high ?: return@forEachIndexed
            val low = bar.low ?: return@forEachIndexed
            val centerX = slotWidth * index + slotWidth / 2f
            val color = if (close >= open) upColor else downColor

            val highY = ((maxPrice - high) / range * chartHeight).toFloat()
            val lowY = ((maxPrice - low) / range * chartHeight).toFloat()
            val openY = ((maxPrice - open) / range * chartHeight).toFloat()
            val closeY = ((maxPrice - close) / range * chartHeight).toFloat()
            val topY = min(openY, closeY)
            val bodyHeight = max(2f, kotlin.math.abs(closeY - openY))

            drawLine(
                color = color,
                start = Offset(centerX, highY),
                end = Offset(centerX, lowY),
                strokeWidth = 2f
            )
            drawRect(
                color = color,
                topLeft = Offset(centerX - bodyWidth / 2f, topY),
                size = Size(bodyWidth, bodyHeight)
            )
        }
    }
}

@Composable
fun IndicatorTrendChart(
    indicators: List<IndicatorSnapshot>,
    modifier: Modifier = Modifier
) {
    val visible = indicators.takeLast(60)
    val closeColor = Color(0xFF2563EB)
    val ma20Color = Color(0xFFF59E0B)
    val ma60Color = Color(0xFF0F9F6E)
    val gridColor = MaterialTheme.colorScheme.outlineVariant

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(240.dp)
    ) {
        val values = visible.flatMap {
            listOfNotNull(it.close, it.ma_20, it.ma_60)
        }.filter { it.isFinite() }
        if (values.size < 2) return@Canvas

        val minValue = values.min()
        val maxValue = values.max()
        val range = (maxValue - minValue).takeIf { it > 0.0 } ?: 1.0

        for (index in 1..4) {
            val y = size.height * index / 5f
            drawLine(
                color = gridColor,
                start = Offset(0f, y),
                end = Offset(size.width, y),
                strokeWidth = 1f
            )
        }

        fun drawSeries(
            series: List<Double?>,
            color: Color
        ) {
            val path = Path()
            var started = false
            series.forEachIndexed { index, value ->
                if (value == null || !value.isFinite()) return@forEachIndexed
                val x = if (series.size <= 1) {
                    size.width / 2f
                } else {
                    size.width * index / (series.size - 1).toFloat()
                }
                val y = ((maxValue - value) / range * size.height).toFloat()
                if (started) {
                    path.lineTo(x, y)
                } else {
                    path.moveTo(x, y)
                    started = true
                }
            }
            if (started) {
                drawPath(
                    path = path,
                    color = color,
                    style = Stroke(width = 2.2f, cap = StrokeCap.Round)
                )
            }
        }

        drawSeries(visible.map { it.close }, closeColor)
        drawSeries(visible.map { it.ma_20 }, ma20Color)
        drawSeries(visible.map { it.ma_60 }, ma60Color)
    }
}
