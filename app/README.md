# Stock Analysis Android Client

这是股票研究软件的 Android 客户端，使用 Kotlin + Jetpack Compose。

## 当前能力

- 底部导航：行情、模型、设置
- 股票代码搜索
- 本地自选股
- 股票详情和最近行情
- K 线、技术指标和最新基本面
- 三 seed 集成模型结果展示
- 后端地址可配置

## 默认后端地址

```text
http://10.0.2.2:8000/
```

Android 模拟器通过 `10.0.2.2` 访问宿主机的 `127.0.0.1`。

## 启动

1. 启动后端：

```powershell
cd D:\DevEcoStudioProjects\gupiao\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

2. 用 Android Studio 打开：

```text
D:\DevEcoStudioProjects\gupiao\app
```

3. 让 Android Studio 自动同步 Gradle 和下载 SDK。

## 下一步客户端功能

- 数据质量页
- 模拟盘信号页
- 指标走势图
