package com.kangtong.stockanalysis.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kangtong.stockanalysis.data.UiState
import com.kangtong.stockanalysis.ui.AppViewModel

@Composable
fun SettingsScreen(viewModel: AppViewModel) {
    val serverState by viewModel.serverState.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("设置", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(16.dp))
        Text("服务器连接", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        when (val state = serverState) {
            is UiState.Idle -> Text("尚未测试")
            is UiState.Loading -> CircularProgressIndicator()
            is UiState.Success -> Text("连接正常")
            is UiState.Error -> Text(
                text = "连接失败：${state.message}",
                color = MaterialTheme.colorScheme.error
            )
        }
        Spacer(Modifier.height(12.dp))
        Button(onClick = viewModel::testServer) {
            Text("测试连接")
        }
        Spacer(Modifier.height(16.dp))
        Text(
            text = "股票数据和预测报告由固定服务器提供。",
            style = MaterialTheme.typography.bodySmall
        )
    }
}
