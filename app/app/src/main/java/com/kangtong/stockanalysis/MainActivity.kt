package com.kangtong.stockanalysis

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.kangtong.stockanalysis.ui.AppViewModel
import com.kangtong.stockanalysis.ui.screens.HomeScreen
import com.kangtong.stockanalysis.ui.screens.ModelScreen
import com.kangtong.stockanalysis.ui.screens.SettingsScreen
import com.kangtong.stockanalysis.ui.screens.StockDetailScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                StockApp()
            }
        }
    }
}

private data class TopTab(
    val route: String,
    val label: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)

@Composable
private fun StockApp() {
    val navController = rememberNavController()
    val viewModel: AppViewModel = viewModel()
    val tabs = listOf(
        TopTab("home", "行情", Icons.Default.Home),
        TopTab("model", "模型", Icons.Default.Info),
        TopTab("settings", "设置", Icons.Default.Settings)
    )
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    Scaffold(
        bottomBar = {
            if (currentRoute in tabs.map { it.route }) {
                NavigationBar {
                    tabs.forEach { tab ->
                        NavigationBarItem(
                            selected = currentRoute == tab.route,
                            onClick = {
                                navController.navigate(tab.route) {
                                    popUpTo(navController.graph.startDestinationId) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = { Icon(tab.icon, contentDescription = tab.label) },
                            label = { Text(tab.label) }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "home",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("home") {
                HomeScreen(viewModel) { item ->
                    viewModel.selectStock(item)
                    navController.navigate("detail/${item.code}")
                }
            }
            composable("detail/{code}") { entry ->
                val code = entry.arguments?.getString("code").orEmpty()
                StockDetailScreen(
                    code = code,
                    viewModel = viewModel,
                    onBack = { navController.popBackStack() }
                )
            }
            composable("model") {
                ModelScreen(viewModel)
            }
            composable("settings") {
                SettingsScreen(viewModel)
            }
        }
    }
}

