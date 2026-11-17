package com.kangtong.stockanalysis.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.kangtong.stockanalysis.data.ApiClient
import com.kangtong.stockanalysis.data.DailyBar
import com.kangtong.stockanalysis.data.FinancialReport
import com.kangtong.stockanalysis.data.IndicatorSnapshot
import com.kangtong.stockanalysis.data.LocalDataStore
import com.kangtong.stockanalysis.data.ModelGroupEnvelope
import com.kangtong.stockanalysis.data.SettingsStore
import com.kangtong.stockanalysis.data.StockApi
import com.kangtong.stockanalysis.data.StockSummary
import com.kangtong.stockanalysis.data.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.format.DateTimeFormatter

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val settings = SettingsStore(application)
    private val localData = LocalDataStore(application)

    private val _watchlist = MutableStateFlow(settings.loadWatchlist())
    val watchlist: StateFlow<List<StockSummary>> = _watchlist.asStateFlow()

    private val _searchResults = MutableStateFlow<List<StockSummary>>(emptyList())
    val searchResults: StateFlow<List<StockSummary>> = _searchResults.asStateFlow()

    private val _searchState = MutableStateFlow<UiState<Unit>>(UiState.Idle)
    val searchState: StateFlow<UiState<Unit>> = _searchState.asStateFlow()

    private val _selectedStock = MutableStateFlow(localData.loadSelectedStock())
    val selectedStock: StateFlow<StockSummary?> = _selectedStock.asStateFlow()

    private val _searchQuery = MutableStateFlow(localData.loadSearchQuery())
    val searchQuery: StateFlow<String> = _searchQuery.asStateFlow()

    private val _dailyState = MutableStateFlow<UiState<List<DailyBar>>>(UiState.Idle)
    val dailyState: StateFlow<UiState<List<DailyBar>>> = _dailyState.asStateFlow()

    private val _indicatorState = MutableStateFlow<UiState<List<IndicatorSnapshot>>>(UiState.Idle)
    val indicatorState: StateFlow<UiState<List<IndicatorSnapshot>>> = _indicatorState.asStateFlow()

    private val _financialState = MutableStateFlow<UiState<FinancialReport>>(UiState.Idle)
    val financialState: StateFlow<UiState<FinancialReport>> = _financialState.asStateFlow()

    private val _modelGroupsState =
        MutableStateFlow<UiState<List<ModelGroupEnvelope>>>(UiState.Idle)
    val modelGroupsState: StateFlow<UiState<List<ModelGroupEnvelope>>> =
        _modelGroupsState.asStateFlow()

    private val _serverState = MutableStateFlow<UiState<Unit>>(UiState.Idle)
    val serverState: StateFlow<UiState<Unit>> = _serverState.asStateFlow()

    private var api: StockApi = ApiClient.create()

    fun testServer() {
        viewModelScope.launch {
            _serverState.value = UiState.Loading
            runCatching { api.health() }
                .onSuccess { _serverState.value = UiState.Success(Unit) }
                .onFailure {
                    _serverState.value = UiState.Error(
                        it.message ?: "服务器连接失败"
                    )
                }
        }
    }

    fun search(query: String) {
        val normalized = query.trim()
        _searchQuery.value = normalized
        localData.saveSearchQuery(normalized)
        viewModelScope.launch {
            _searchState.value = UiState.Loading
            runCatching { api.search(normalized) }
                .onSuccess {
                    _searchResults.value = it
                    _searchState.value = UiState.Success(Unit)
                }
                .onFailure { _searchState.value = UiState.Error(it.message ?: "网络请求失败") }
        }
    }

    fun setSearchInput(value: String) {
        _searchQuery.value = value
    }

    fun selectStock(item: StockSummary) {
        _selectedStock.value = item
        localData.saveSelectedStock(item)
    }

    fun loadDaily(code: String) {
        viewModelScope.launch {
            _dailyState.value = UiState.Loading
            val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
            val end = LocalDate.now().format(formatter)
            val start = LocalDate.now().minusDays(365).format(formatter)
            runCatching { api.dailyBars(code, start, end) }
                .onSuccess { _dailyState.value = UiState.Success(it.sortedBy { row -> row.trade_date }) }
                .onFailure { _dailyState.value = UiState.Error(it.message ?: "行情加载失败") }
        }
    }

    fun loadIndicators(code: String) {
        viewModelScope.launch {
            _indicatorState.value = UiState.Loading
            val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
            val end = LocalDate.now().format(formatter)
            val start = LocalDate.now().minusDays(365).format(formatter)
            runCatching { api.indicators(code, start, end) }
                .onSuccess { rows ->
                    _indicatorState.value = UiState.Success(rows.sortedBy { row -> row.trade_date })
                }
                .onFailure { _indicatorState.value = UiState.Error(it.message ?: "指标加载失败") }
        }
    }

    fun loadFinancialReport(code: String) {
        viewModelScope.launch {
            _financialState.value = UiState.Loading
            runCatching { api.latestFinancialReport(code) }
                .onSuccess { _financialState.value = UiState.Success(it) }
                .onFailure { _financialState.value = UiState.Error(it.message ?: "财务数据加载失败") }
        }
    }

    fun loadModelGroups() {
        val cached = localData.loadModelGroups()
        if (cached != null) {
            _modelGroupsState.value = UiState.Success(cached)
        } else {
            _modelGroupsState.value = UiState.Loading
        }
        viewModelScope.launch {
            runCatching { api.modelGroups() }
                .onSuccess {
                    localData.saveModelGroups(it)
                    _modelGroupsState.value = UiState.Success(it)
                }
                .onFailure {
                    if (cached == null) {
                        _modelGroupsState.value = UiState.Error(
                            it.message ?: "模型组加载失败"
                        )
                    }
                }
        }
    }

    fun addToWatchlist(item: StockSummary) {
        val current = _watchlist.value
        if (current.any { it.code == item.code }) return
        val next = current + item
        settings.saveWatchlist(next)
        _watchlist.value = next
    }

    fun removeFromWatchlist(code: String) {
        val next = _watchlist.value.filterNot { it.code == code }
        settings.saveWatchlist(next)
        _watchlist.value = next
    }
}
