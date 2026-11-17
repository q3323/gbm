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
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kangtong.stockanalysis.data.GroupEvaluation
import com.kangtong.stockanalysis.data.GroupRanking
import com.kangtong.stockanalysis.data.ModelGroupEnvelope
import com.kangtong.stockanalysis.data.UiState
import com.kangtong.stockanalysis.ui.AppViewModel
import java.text.DecimalFormat

@Composable
fun ModelScreen(viewModel: AppViewModel) {
    val state by viewModel.modelGroupsState.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        viewModel.loadModelGroups()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("分组模型预测", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text(
            text = "同模型 5-seed Bagging · 组内连续分数排名",
            style = MaterialTheme.typography.bodyMedium
        )
        Spacer(Modifier.height(8.dp))
        TextButton(onClick = viewModel::loadModelGroups) {
            Text("刷新结果")
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            when (val current = state) {
                is UiState.Idle -> {}
                is UiState.Loading -> Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator()
                    Spacer(Modifier.width(12.dp))
                    Text("加载模型组...")
                }
                is UiState.Error -> Text(
                    text = current.message,
                    color = MaterialTheme.colorScheme.error
                )
                is UiState.Success -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(current.data, key = { it.id }) { group ->
                            ModelGroupCard(group)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelGroupCard(group: ModelGroupEnvelope) {
    val report = group.report
    val prediction = report?.group_prediction
    val metadata = report?.metadata

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(group.name, style = MaterialTheme.typography.titleLarge)
                    Text(
                        text = when (group.market) {
                            "US" -> "美股"
                            "CN" -> "A 股"
                            else -> group.market ?: "-"
                        },
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Text(
                    text = directionLabel(prediction?.direction),
                    color = signalColor(prediction?.direction),
                    style = MaterialTheme.typography.titleMedium
                )
            }

            if (group.audit_status == "fail") {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "严谨审计未通过，结果仅供研究",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium
                )
            } else if (group.audit_status == "pass") {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "严谨审计通过",
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            if (group.status != "ready" || report == null) {
                Spacer(Modifier.height(12.dp))
                Text("模型尚未训练", color = MaterialTheme.colorScheme.error)
                return@Column
            }

            Spacer(Modifier.height(12.dp))
            Text("平均预期收益: ${formatSignedPercent(prediction?.expected_return)}")
            Text("校准前概率: ${formatPercent(prediction?.raw_up_probability)}")
            Text("校准后概率: ${formatPercent(prediction?.up_probability)}")
            Text("概率偏移: ${formatSignedPercent(prediction?.calibration_shift)}")
            Text("种子间收益分歧: ${formatPercent(prediction?.prediction_std)}")
            Text("趋势合格比例: ${formatPercent(prediction?.trend_eligible_ratio)}")
            Text("可交易信号数: ${prediction?.trade_signal_count ?: 0}")
            Text("预测生成: ${metadata?.generated_on ?: "-"}")
            Text("数据截止: ${metadata?.as_of_date ?: "-"}")
            Text("Seeds: ${metadata?.seeds?.joinToString().orEmpty()}")

            report.evaluation?.let { evaluation ->
                Spacer(Modifier.height(12.dp))
                HorizontalDivider()
                Spacer(Modifier.height(8.dp))
                Text("独立评估", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                GroupEvaluationRows(evaluation)
            }

            Spacer(Modifier.height(12.dp))
            HorizontalDivider()
            Spacer(Modifier.height(8.dp))
            Text("组内排名", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))

            report.rankings.forEachIndexed { index, ranking ->
                if (index > 0) {
                    HorizontalDivider(
                        modifier = Modifier.padding(vertical = 6.dp)
                    )
                }
                GroupRankingRow(ranking)
            }

            if (report.supplemental_products.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                HorizontalDivider()
                Spacer(Modifier.height(8.dp))
                Text(
                    "其他纳指 ETF",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "包含杠杆和反向产品，不参与横截面评分",
                    style = MaterialTheme.typography.bodySmall
                )
                report.supplemental_products.forEach { product ->
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "${product.stock_name ?: product.stock_code} · " +
                            "${product.stock_code} · " +
                            supplementalCategoryLabel(product),
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        "产品趋势 ${directionLabel(product.direction)} · " +
                            "趋势分 ${formatPercent(product.trend_score)} · " +
                            "5日 ${formatSignedPercent(product.return_5d)} · " +
                            "20日 ${formatSignedPercent(product.return_20d)}",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

private fun supplementalCategoryLabel(
    product: com.kangtong.stockanalysis.data.SupplementalProduct
): String {
    val category = when (product.category) {
        "leveraged_long" -> "杠杆做多"
        "inverse" -> "反向产品"
        else -> "1倍做多"
    }
    val leverage = product.leverage?.let { "${it}x" } ?: ""
    return listOf(category, leverage).filter { it.isNotBlank() }.joinToString(" ")
}

@Composable
private fun GroupEvaluationRows(evaluation: GroupEvaluation) {
    val rankIcLabel = if (evaluation.rank_ic_mode == "time_series") {
        "时序 Rank IC"
    } else {
        "Rank IC"
    }
    Text("$rankIcLabel: ${formatMetric(evaluation.rank_ic_mean)}")
    Text("Rank IC 标准差: ${formatMetric(evaluation.rank_ic_std)}")
    Text("ICIR: ${formatMetric(evaluation.icir)}")
    Text("IC > 0 比例: ${formatPercent(evaluation.positive_rank_ic_rate)}")
    Text("原始概率 Brier: ${formatMetric(evaluation.raw_brier_score)}")
    Text("校准概率 Brier: ${formatMetric(evaluation.calibrated_brier_score)}")
    Text("平均概率偏移: ${formatSignedPercent(evaluation.mean_calibration_shift)}")
    evaluation.signal_correlation?.let {
        Text("时序信号相关性: ${formatMetric(it)}")
    }
    Text("策略总收益: ${formatSignedPercent(evaluation.total_return)}")
    Text("Sharpe: ${formatMetric(evaluation.sharpe)}")
    Text("最大回撤: ${formatPercent(evaluation.max_drawdown)}")
    Text("方向命中率: ${formatPercent(evaluation.hit_rate)}")
    Text("趋势过滤收益: ${formatSignedPercent(evaluation.trend_filtered_total_return)}")
    Text("趋势过滤回撤: ${formatPercent(evaluation.trend_filtered_max_drawdown)}")
    Text("趋势过滤调仓次数: ${evaluation.trend_filtered_periods ?: 0}")
    evaluation.baselines?.let { baselines ->
        Text("动量 Rank IC: ${formatMetric(baselines.momentum_rank_ic)}")
        Text("逻辑回归 Rank IC: ${formatMetric(baselines.logistic_rank_ic)}")
        Text(
            text = if (baselines.gbm_beats_momentum == true) {
                "GBM 优于动量基线"
            } else {
                "GBM 未优于动量基线"
            }
        )
        Text(
            text = if (baselines.gbm_beats_logistic == true) {
                "GBM 优于逻辑回归"
            } else {
                "GBM 未优于逻辑回归"
            }
        )
    }
}

@Composable
private fun GroupRankingRow(ranking: GroupRanking) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "#${ranking.rank ?: "-"}",
            modifier = Modifier.width(48.dp),
            style = MaterialTheme.typography.labelLarge
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = ranking.stock_name ?: ranking.stock_code,
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = ranking.stock_code,
                style = MaterialTheme.typography.bodySmall
            )
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(
                text = "分数 ${formatScore(ranking.final_score ?: ranking.rank_score)}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = formatSignedPercent(ranking.expected_return),
                color = signalColor(ranking.direction),
                style = MaterialTheme.typography.bodyMedium
            )
            if (ranking.trade_signal == true) {
                Text(
                    text = "可交易",
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }
    }
}

@Composable
private fun signalColor(direction: String?): Color {
    return when (direction) {
        "up" -> MaterialTheme.colorScheme.primary
        "down" -> MaterialTheme.colorScheme.error
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
}

private fun directionLabel(direction: String?): String {
    return when (direction) {
        "up" -> "偏多"
        "down" -> "偏空"
        "mixed" -> "分歧"
        else -> "-"
    }
}

private fun formatPercent(value: Double?): String {
    return value?.let { DecimalFormat("0.00%").format(it) } ?: "-"
}

private fun formatSignedPercent(value: Double?): String {
    return value?.let {
        val prefix = if (it > 0) "+" else ""
        prefix + DecimalFormat("0.00%").format(it)
    } ?: "-"
}

private fun formatScore(value: Double?): String {
    return value?.let { DecimalFormat("0.0").format(it) } ?: "-"
}

private fun formatMetric(value: Double?): String {
    return value?.let { DecimalFormat("0.0000").format(it) } ?: "-"
}
