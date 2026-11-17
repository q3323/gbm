package com.kangtong.stockanalysis.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kangtong.stockanalysis.data.DailyBar
import com.kangtong.stockanalysis.data.FinancialReport
import com.kangtong.stockanalysis.data.IndicatorSnapshot
import com.kangtong.stockanalysis.data.StockSummary
import com.kangtong.stockanalysis.data.UiState
import com.kangtong.stockanalysis.ui.AppViewModel
import java.text.DecimalFormat

private val stockDetailTabs = listOf("行情", "分析", "指标", "基本面")
private val fundDetailTabs = listOf("行情", "分析", "指标", "基金信息")

@Composable
fun StockDetailScreen(
    code: String,
    viewModel: AppViewModel,
    onBack: () -> Unit
) {
    val selected by viewModel.selectedStock.collectAsStateWithLifecycle()
    val dailyState by viewModel.dailyState.collectAsStateWithLifecycle()
    val indicatorState by viewModel.indicatorState.collectAsStateWithLifecycle()
    val financialState by viewModel.financialState.collectAsStateWithLifecycle()
    val stock = selected ?: StockSummary(
        code = code,
        name = code,
        exchange = "",
        source = ""
    )
    val isFund = stock.asset_type == "fund"
    val detailTabs = if (isFund) fundDetailTabs else stockDetailTabs
    var selectedTab by rememberSaveable(code, isFund) {
        mutableIntStateOf(if (isFund) 1 else 0)
    }

    LaunchedEffect(code, isFund) {
        viewModel.loadDaily(code)
        viewModel.loadIndicators(code)
        if (!isFund) {
            viewModel.loadFinancialReport(code)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "返回"
                )
            }
            Column {
                Text(stock.name, style = MaterialTheme.typography.headlineSmall)
                Text(
                    text = buildString {
                        append(stock.code)
                        append(" · ")
                        append(stock.exchange)
                        append(" · ")
                        append(if (isFund) "基金" else "股票")
                    },
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
        Spacer(Modifier.height(12.dp))

        TabRow(selectedTabIndex = selectedTab) {
            detailTabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title) }
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            when (selectedTab) {
                0 -> MarketTab(dailyState = dailyState)
                1 -> AnalysisTab(
                    dailyState = dailyState,
                    indicatorState = indicatorState
                )
                2 -> IndicatorsTab(indicatorState = indicatorState)
                else -> if (isFund) {
                    FundInfoTab(
                        dailyState = dailyState,
                        indicatorState = indicatorState
                    )
                } else {
                    FundamentalsTab(financialState = financialState)
                }
            }
        }
    }
}

@Composable
private fun MarketTab(
    dailyState: UiState<List<DailyBar>>
) {
    when (val state = dailyState) {
        is UiState.Loading -> LoadingRow("加载行情...")
        is UiState.Error -> ErrorText(state.message)
        is UiState.Success -> {
            val bars = state.data
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (bars.isEmpty()) {
                    item { Text("暂无行情数据") }
                } else {
                    val latest = bars.last()
                    item {
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text("最新收盘", style = MaterialTheme.typography.labelMedium)
                                Text(
                                    text = formatPrice(latest.close),
                                    style = MaterialTheme.typography.headlineSmall
                                )
                                latest.pct_change?.let { change ->
                                    Text(
                                        text = "${formatSigned(change)}%",
                                        color = if (change >= 0) {
                                            MaterialTheme.colorScheme.primary
                                        } else {
                                            MaterialTheme.colorScheme.error
                                        },
                                        style = MaterialTheme.typography.titleMedium
                                    )
                                }
                            }
                        }
                    }
                    item {
                        Text("近期 K 线", style = MaterialTheme.typography.titleMedium)
                    }
                    item {
                        CandlestickChart(bars = bars.takeLast(90))
                    }
                    item {
                        Text("最近交易", style = MaterialTheme.typography.titleMedium)
                    }
                    items(bars.takeLast(20).reversed(), key = { it.trade_date }) { bar ->
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp)
                            ) {
                                Text(
                                    text = bar.trade_date,
                                    modifier = Modifier.weight(1.2f)
                                )
                                Text(
                                    text = formatPrice(bar.close),
                                    modifier = Modifier.weight(0.8f)
                                )
                                Text(
                                    text = formatPercentPoints(bar.pct_change),
                                    modifier = Modifier.weight(0.8f)
                                )
                            }
                        }
                    }
                }
            }
        }
        else -> {}
    }
}

private data class DailyAnalysis(
    val return5d: Double?,
    val return20d: Double?,
    val return60d: Double?,
    val return120d: Double?,
    val return240d: Double?,
    val volatility20d: Double?,
    val volatility60d: Double?,
    val maxDrawdown120d: Double?,
    val maxDrawdown240d: Double?,
    val positiveRate60d: Double?,
    val averageAmplitude20d: Double?,
    val averageVolume20d: Double?,
    val averageVolume60d: Double?,
    val averageAmount20d: Double?,
    val averageAmount60d: Double?,
    val latestVolumeRatio20d: Double?,
    val high240d: Double?,
    val low240d: Double?,
    val distanceFromHigh240d: Double?
)

@Composable
private fun AnalysisTab(
    dailyState: UiState<List<DailyBar>>,
    indicatorState: UiState<List<IndicatorSnapshot>>
) {
    when (val state = dailyState) {
        is UiState.Loading -> LoadingRow("计算分析数据...")
        is UiState.Error -> ErrorText(state.message)
        is UiState.Success -> {
            val bars = state.data
            if (bars.isEmpty()) {
                Text("暂无可分析数据")
                return
            }

            val analysis = analyzeDailyBars(bars)
            val latest = bars.last()
            val latestIndicator = (indicatorState as? UiState.Success)?.data?.lastOrNull()

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    IndicatorCard(title = "区间概览") {
                        IndicatorRow("最新收盘", formatPrice(latest.close))
                        IndicatorRow("近 20 日收益", formatPercent(analysis.return20d))
                        IndicatorRow("近 60 日收益", formatPercent(analysis.return60d))
                        IndicatorRow("近 120 日收益", formatPercent(analysis.return120d))
                        IndicatorRow("近 240 日收益", formatPercent(analysis.return240d))
                        IndicatorRow("近 60 日上涨比例", formatPercent(analysis.positiveRate60d))
                    }
                }
                item {
                    IndicatorCard(title = "多周期收益") {
                        IndicatorRow("近 5 个交易日", formatPercent(analysis.return5d))
                        IndicatorRow("近 20 个交易日", formatPercent(analysis.return20d))
                        IndicatorRow("近 60 个交易日", formatPercent(analysis.return60d))
                        IndicatorRow("近 120 个交易日", formatPercent(analysis.return120d))
                        IndicatorRow("近 240 个交易日", formatPercent(analysis.return240d))
                    }
                }
                item {
                    IndicatorCard(title = "风险与回撤") {
                        IndicatorRow("20 日年化波动", formatPercent(analysis.volatility20d))
                        IndicatorRow("60 日年化波动", formatPercent(analysis.volatility60d))
                        IndicatorRow("120 日最大回撤", formatPercent(analysis.maxDrawdown120d))
                        IndicatorRow("240 日最大回撤", formatPercent(analysis.maxDrawdown240d))
                        IndicatorRow("距 240 日高点", formatPercent(analysis.distanceFromHigh240d))
                        IndicatorRow("240 日最高", formatPrice(analysis.high240d))
                        IndicatorRow("240 日最低", formatPrice(analysis.low240d))
                    }
                }
                item {
                    IndicatorCard(title = "成交量状态") {
                        IndicatorRow("最新量能比", formatRatio(analysis.latestVolumeRatio20d))
                        IndicatorRow("20 日平均成交量", formatAmount(analysis.averageVolume20d))
                        IndicatorRow("60 日平均成交量", formatAmount(analysis.averageVolume60d))
                        IndicatorRow("20 日平均成交额", formatAmount(analysis.averageAmount20d))
                        IndicatorRow("60 日平均成交额", formatAmount(analysis.averageAmount60d))
                        IndicatorRow("20 日平均振幅", formatPercent(analysis.averageAmplitude20d))
                    }
                }
                item {
                    IndicatorCard(title = "趋势与动量") {
                        IndicatorRow(
                            "均线结构",
                            movingAverageState(latestIndicator)
                        )
                        IndicatorRow(
                            "RSI14",
                            formatIndicatorState(
                                latestIndicator?.rsi_14,
                                rsiState(latestIndicator?.rsi_14)
                            )
                        )
                        IndicatorRow(
                            "MACD 柱",
                            formatIndicatorState(
                                latestIndicator?.macd_histogram,
                                macdState(latestIndicator?.macd_histogram)
                            )
                        )
                        IndicatorRow(
                            "KDJ J 值",
                            formatIndicatorState(
                                latestIndicator?.kdj_j,
                                kdjState(latestIndicator?.kdj_j)
                            )
                        )
                    }
                }
                item {
                    IndicatorCard(title = "数据覆盖") {
                        IndicatorRow("交易记录", "${bars.size} 条")
                        IndicatorRow("起始日期", bars.first().trade_date)
                        IndicatorRow("截止日期", latest.trade_date)
                        IndicatorRow("复权方式", latest.adjust_type)
                        IndicatorRow("数据来源", latest.source)
                    }
                }
            }
        }
        else -> {}
    }
}

private fun analyzeDailyBars(bars: List<DailyBar>): DailyAnalysis {
    val returns20 = dailyReturns(bars.takeLast(21))
    val returns60 = dailyReturns(bars.takeLast(61))
    val last60Returns = returns60.takeLast(60)
    val recent20 = bars.takeLast(20)
    val previous20 = bars.dropLast(1).takeLast(20)
    val recent120 = bars.takeLast(120)
    val recent240 = bars.takeLast(240)
    val previous60 = bars.dropLast(1).takeLast(60)
    val latest = bars.last()

    val averageVolume20 = previous20.mapNotNull { it.volume }
        .takeIf { it.isNotEmpty() }
        ?.average()
    val averageVolume60 = previous60.mapNotNull { it.volume }
        .takeIf { it.isNotEmpty() }
        ?.average()
    val high240 = recent240.mapNotNull { it.high }.maxOrNull()
    val low240 = recent240.mapNotNull { it.low }.minOrNull()

    return DailyAnalysis(
        return5d = periodReturn(bars, 5),
        return20d = periodReturn(bars, 20),
        return60d = periodReturn(bars, 60),
        return120d = periodReturn(bars, 120),
        return240d = periodReturn(bars, 240),
        volatility20d = annualizedVolatility(returns20.takeLast(20)),
        volatility60d = annualizedVolatility(returns60.takeLast(60)),
        maxDrawdown120d = maxDrawdown(recent120),
        maxDrawdown240d = maxDrawdown(recent240),
        positiveRate60d = last60Returns.takeIf { it.isNotEmpty() }
            ?.let { values -> values.count { it > 0 }.toDouble() / values.size },
        averageAmplitude20d = recent20.mapNotNull { bar ->
            val high = bar.high
            val low = bar.low
            val base = bar.pre_close ?: latest.close
            if (high != null && low != null && base != null && base > 0) {
                (high - low) / base
            } else {
                null
            }
        }.takeIf { it.isNotEmpty() }?.average(),
        averageVolume20d = averageVolume20,
        averageVolume60d = averageVolume60,
        averageAmount20d = previous20.mapNotNull { it.amount }
            .takeIf { it.isNotEmpty() }
            ?.average(),
        averageAmount60d = previous60.mapNotNull { it.amount }
            .takeIf { it.isNotEmpty() }
            ?.average(),
        latestVolumeRatio20d = if (
            latest.volume != null &&
            averageVolume20 != null &&
            averageVolume20 > 0
        ) {
            latest.volume / averageVolume20
        } else {
            null
        },
        high240d = high240,
        low240d = low240,
        distanceFromHigh240d = if (
            latest.close != null &&
            high240 != null &&
            high240 > 0
        ) {
            latest.close / high240 - 1.0
        } else {
            null
        }
    )
}

private fun periodReturn(bars: List<DailyBar>, periods: Int): Double? {
    if (bars.size <= periods) return null
    val start = bars[bars.lastIndex - periods].close ?: return null
    val end = bars.last().close ?: return null
    if (start == 0.0) return null
    return end / start - 1.0
}

private fun dailyReturns(bars: List<DailyBar>): List<Double> {
    return bars.zipWithNext().mapNotNull { (previous, current) ->
        val start = previous.close
        val end = current.close
        if (start != null && end != null && start > 0) {
            end / start - 1.0
        } else {
            null
        }
    }
}

private fun annualizedVolatility(returns: List<Double>): Double? {
    if (returns.size < 2) return null
    val mean = returns.average()
    val variance = returns.sumOf { value ->
        val difference = value - mean
        difference * difference
    } / returns.size
    return kotlin.math.sqrt(variance) * kotlin.math.sqrt(252.0)
}

private fun maxDrawdown(bars: List<DailyBar>): Double? {
    val closes = bars.mapNotNull { it.close }
    if (closes.size < 2) return null
    var peak = closes.first()
    var worst = 0.0
    closes.forEach { value ->
        peak = kotlin.math.max(peak, value)
        if (peak > 0) {
            worst = kotlin.math.min(worst, value / peak - 1.0)
        }
    }
    return worst
}

private fun movingAverageState(latest: IndicatorSnapshot?): String {
    val close = latest?.close ?: return "指标数据未就绪"
    val ma5 = latest.ma_5 ?: return "指标数据未就绪"
    val ma20 = latest.ma_20 ?: return "指标数据未就绪"
    val ma60 = latest.ma_60 ?: return "指标数据未就绪"
    return when {
        close > ma5 && ma5 > ma20 && ma20 > ma60 -> "多头排列"
        close < ma5 && ma5 < ma20 && ma20 < ma60 -> "空头排列"
        else -> "均线交织"
    }
}

private fun rsiState(value: Double?): String = when {
    value == null -> "未就绪"
    value >= 70 -> "偏热"
    value <= 30 -> "偏冷"
    else -> "中性"
}

private fun macdState(value: Double?): String = when {
    value == null -> "未就绪"
    value > 0 -> "红柱"
    value < 0 -> "绿柱"
    else -> "零轴"
}

private fun kdjState(value: Double?): String = when {
    value == null -> "未就绪"
    value >= 80 -> "偏高"
    value <= 20 -> "偏低"
    else -> "正常"
}

private fun formatIndicatorState(value: Double?, state: String): String {
    return if (value == null) state else "${formatValue(value)} · $state"
}

private fun formatRatio(value: Double?): String {
    return value?.let { DecimalFormat("0.00").format(it) + "x" } ?: "-"
}

@Composable
private fun IndicatorsTab(
    indicatorState: UiState<List<IndicatorSnapshot>>
) {
    when (val state = indicatorState) {
        is UiState.Loading -> LoadingRow("加载技术指标...")
        is UiState.Error -> ErrorText(state.message)
        is UiState.Success -> {
            val latest = state.data.lastOrNull()
            if (latest == null) {
                Text("暂无技术指标")
                return
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    IndicatorCard(title = "价格趋势") {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Text("收盘", style = MaterialTheme.typography.labelSmall)
                            Text("MA20", style = MaterialTheme.typography.labelSmall)
                            Text("MA60", style = MaterialTheme.typography.labelSmall)
                        }
                        Spacer(Modifier.height(8.dp))
                        IndicatorTrendChart(indicators = state.data)
                    }
                }
                item {
                    IndicatorCard(title = "价格与均线") {
                        IndicatorRow("收盘价", formatPrice(latest.close))
                        IndicatorRow("MA5", formatPrice(latest.ma_5))
                        IndicatorRow("MA10", formatPrice(latest.ma_10))
                        IndicatorRow("MA20", formatPrice(latest.ma_20))
                        IndicatorRow("MA60", formatPrice(latest.ma_60))
                    }
                }
                item {
                    IndicatorCard(title = "MACD") {
                        IndicatorRow("DIF", formatSigned(latest.macd))
                        IndicatorRow("DEA", formatSigned(latest.macd_signal))
                        IndicatorRow("柱线", formatSigned(latest.macd_histogram))
                    }
                }
                item {
                    IndicatorCard(title = "强弱与随机指标") {
                        IndicatorRow("RSI14", formatValue(latest.rsi_14))
                        IndicatorRow("K", formatValue(latest.kdj_k))
                        IndicatorRow("D", formatValue(latest.kdj_d))
                        IndicatorRow("J", formatValue(latest.kdj_j))
                    }
                }
                item {
                    IndicatorCard(title = "波动与通道") {
                        IndicatorRow("布林上轨", formatPrice(latest.boll_upper_20))
                        IndicatorRow("布林中轨", formatPrice(latest.boll_mid_20))
                        IndicatorRow("布林下轨", formatPrice(latest.boll_lower_20))
                        IndicatorRow("ATR14", formatValue(latest.atr_14))
                        IndicatorRow("年化波动", formatPercent(latest.volatility_20))
                    }
                }
            }
        }
        else -> {}
    }
}

@Composable
private fun FundamentalsTab(
    financialState: UiState<FinancialReport>
) {
    when (val state = financialState) {
        is UiState.Loading -> LoadingRow("加载财务数据...")
        is UiState.Error -> ErrorText(state.message)
        is UiState.Success -> {
            val report = state.data
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text("报告期", style = MaterialTheme.typography.titleMedium)
                            Text(report.report_date)
                            Text(
                                text = "公告日: ${report.announced_date ?: "未提供"}",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
                item {
                    IndicatorCard(title = "盈利能力") {
                        IndicatorRow("ROE", formatPercent(report.roe))
                        IndicatorRow("净利率", formatPercent(report.net_margin))
                        IndicatorRow("毛利率", formatPercent(report.gross_margin))
                        IndicatorRow("EPS", formatValue(report.eps))
                    }
                }
                item {
                    IndicatorCard(title = "成长性") {
                        IndicatorRow("净利润同比", formatPercent(report.net_profit_yoy))
                        IndicatorRow("EPS 同比", formatPercent(report.eps_yoy))
                        IndicatorRow("净资产同比", formatPercent(report.equity_yoy))
                    }
                }
                item {
                    IndicatorCard(title = "规模与质量") {
                        IndicatorRow("营业收入", formatAmount(report.revenue))
                        IndicatorRow("净利润", formatAmount(report.net_profit))
                        IndicatorRow("资产负债率", formatPercent(report.debt_ratio))
                        IndicatorRow("资产周转率", formatValue(report.asset_turnover))
                        IndicatorRow("经营现金流 / 营收", formatValue(report.cfo_to_revenue))
                    }
                }
            }
        }
        else -> {}
    }
}

@Composable
private fun FundInfoTab(
    dailyState: UiState<List<DailyBar>>,
    indicatorState: UiState<List<IndicatorSnapshot>>
) {
    when (val state = dailyState) {
        is UiState.Loading -> LoadingRow("加载基金数据...")
        is UiState.Error -> ErrorText(state.message)
        is UiState.Success -> {
            val bars = state.data
            if (bars.isEmpty()) {
                Text("暂无基金数据")
                return
            }
            val analysis = analyzeDailyBars(bars)
            val latest = bars.last()
            val latestIndicator =
                (indicatorState as? UiState.Success)?.data?.lastOrNull()

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    IndicatorCard(title = "ETF 概览") {
                        IndicatorRow("产品类型", "交易所交易基金")
                        IndicatorRow("最新收盘", formatPrice(latest.close))
                        IndicatorRow("交易日期", latest.trade_date)
                        IndicatorRow("数据来源", latest.source)
                        IndicatorRow("交易记录", "${bars.size} 条")
                    }
                }
                item {
                    IndicatorCard(title = "区间表现") {
                        IndicatorRow("近 20 日收益", formatPercent(analysis.return20d))
                        IndicatorRow("近 60 日收益", formatPercent(analysis.return60d))
                        IndicatorRow("近 120 日收益", formatPercent(analysis.return120d))
                        IndicatorRow("近 240 日收益", formatPercent(analysis.return240d))
                        IndicatorRow("近 60 日上涨比例", formatPercent(analysis.positiveRate60d))
                    }
                }
                item {
                    IndicatorCard(title = "风险与流动性") {
                        IndicatorRow("20 日年化波动", formatPercent(analysis.volatility20d))
                        IndicatorRow("60 日年化波动", formatPercent(analysis.volatility60d))
                        IndicatorRow("120 日最大回撤", formatPercent(analysis.maxDrawdown120d))
                        IndicatorRow("240 日最大回撤", formatPercent(analysis.maxDrawdown240d))
                        IndicatorRow("20 日平均成交额", formatAmount(analysis.averageAmount20d))
                        IndicatorRow("60 日平均成交额", formatAmount(analysis.averageAmount60d))
                        IndicatorRow("最新量能比", formatRatio(analysis.latestVolumeRatio20d))
                    }
                }
                item {
                    IndicatorCard(title = "技术状态") {
                        IndicatorRow("均线结构", movingAverageState(latestIndicator))
                        IndicatorRow(
                            "RSI14",
                            formatIndicatorState(
                                latestIndicator?.rsi_14,
                                rsiState(latestIndicator?.rsi_14)
                            )
                        )
                        IndicatorRow(
                            "MACD 柱",
                            formatIndicatorState(
                                latestIndicator?.macd_histogram,
                                macdState(latestIndicator?.macd_histogram)
                            )
                        )
                    }
                }
            }
        }
        else -> {}
    }
}

@Composable
private fun LoadingRow(text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator()
        Spacer(Modifier.width(12.dp))
        Text(text)
    }
}

@Composable
private fun ErrorText(message: String) {
    Text(
        text = message,
        color = MaterialTheme.colorScheme.error
    )
}

@Composable
private fun IndicatorCard(
    title: String,
    content: @Composable () -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun IndicatorRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyMedium
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

private fun formatPrice(value: Double?): String {
    return value?.let { DecimalFormat("#,##0.00").format(it) } ?: "-"
}

private fun formatValue(value: Double?): String {
    return value?.let { DecimalFormat("0.00").format(it) } ?: "-"
}

private fun formatSigned(value: Double?): String {
    return value?.let {
        val prefix = if (it > 0) "+" else ""
        prefix + DecimalFormat("0.00").format(it)
    } ?: "-"
}

private fun formatPercent(value: Double?): String {
    return value?.let { DecimalFormat("0.00%").format(it) } ?: "-"
}

private fun formatPercentPoints(value: Double?): String {
    return value?.let { "${formatSigned(it)}%" } ?: "-"
}

private fun formatAmount(value: Double?): String {
    if (value == null) return "-"
    val absolute = kotlin.math.abs(value)
    return when {
        absolute >= 100_000_000.0 -> DecimalFormat("0.00亿").format(value / 100_000_000.0)
        absolute >= 10_000.0 -> DecimalFormat("0.00万").format(value / 10_000.0)
        else -> DecimalFormat("#,##0.00").format(value)
    }
}
