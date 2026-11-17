package com.kangtong.stockanalysis.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kangtong.stockanalysis.data.StockSummary
import com.kangtong.stockanalysis.data.UiState
import com.kangtong.stockanalysis.ui.AppViewModel

@Composable
fun HomeScreen(
    viewModel: AppViewModel,
    onOpenStock: (StockSummary) -> Unit
) {
    val queryText by viewModel.searchQuery.collectAsStateWithLifecycle()
    val watchlist by viewModel.watchlist.collectAsStateWithLifecycle()
    val searchResults by viewModel.searchResults.collectAsStateWithLifecycle()
    val searchState by viewModel.searchState.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "股票研究",
            style = MaterialTheme.typography.headlineMedium
        )
        Spacer(Modifier.height(16.dp))
        OutlinedTextField(
            value = queryText,
            onValueChange = viewModel::setSearchInput,
            label = { Text("股票或基金代码") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(8.dp))
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(onClick = { viewModel.search(queryText) }) {
                Text("搜索")
            }
            Spacer(Modifier.width(12.dp))
            when (val state = searchState) {
                is UiState.Loading -> CircularProgressIndicator()
                is UiState.Error -> Text(
                    text = state.message,
                    color = MaterialTheme.colorScheme.error
                )
                else -> {}
            }
        }
        Spacer(Modifier.height(16.dp))
        Text("搜索结果", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (searchResults.isNotEmpty()) {
                items(searchResults, key = { it.code }) { item ->
                    StockRow(
                        item = item,
                        inWatchlist = watchlist.any { it.code == item.code },
                        onClick = { onOpenStock(item) },
                        onToggleWatch = {
                            if (watchlist.any { it.code == item.code }) {
                                viewModel.removeFromWatchlist(item.code)
                            } else {
                                viewModel.addToWatchlist(item)
                            }
                        }
                    )
                }
            }
            item {
                Text(
                    text = "自选列表",
                    style = MaterialTheme.typography.titleMedium
                )
            }
            items(watchlist, key = { "watch_${it.code}" }) { item ->
                StockRow(
                    item = item,
                    inWatchlist = true,
                    onClick = { onOpenStock(item) },
                    onToggleWatch = { viewModel.removeFromWatchlist(item.code) }
                )
            }
        }
    }
}

@Composable
private fun StockRow(
    item: StockSummary,
    inWatchlist: Boolean,
    onClick: () -> Unit,
    onToggleWatch: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(item.name, style = MaterialTheme.typography.titleMedium)
                val typeLabel =
                    if (item.asset_type == "fund") "基金" else "股票"
                Text(
                    text = "${item.code} · ${item.exchange} · $typeLabel · ${item.source}",
                    style = MaterialTheme.typography.bodySmall
                )
            }
            TextButton(onClick = onToggleWatch) {
                Text(if (inWatchlist) "移除" else "加自选")
            }
        }
    }
}

