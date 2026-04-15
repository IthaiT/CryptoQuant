# CryptoQuant

这个仓库现在不是一个单线项目，而是两条线并存：

1. 当前主线：[`app.py`](./app.py) 是一个 Streamlit 蒙特卡洛个人财务规划器。
2. 历史副线：`src/`、`script/`、`notebooks/` 里保留了一套加密货币量化研究/回测工具链。

仓库名还是 `CryptoQuant`，但它已经不能准确代表当前主入口。

## 先看什么

- 想最快理解“现在这个项目在做什么”：看 [`app.py`](./app.py) 和 [`docs/tutorials/01-快速开始指南.md`](./docs/tutorials/01-快速开始指南.md)
- 想看早期量化研究代码：看 [`docs/tutorials/02-数据下载指南.md`](./docs/tutorials/02-数据下载指南.md) 和 [`docs/tutorials/03-回测框架手册.md`](./docs/tutorials/03-回测框架手册.md)
- 想看 notebooks 在研究什么：看 [`docs/tutorials/04-策略开发教程.md`](./docs/tutorials/04-策略开发教程.md)
- 想知道哪些文件已经脱节、哪些只是知识库：看 [`docs/tutorials/05-进阶功能.md`](./docs/tutorials/05-进阶功能.md)

## 当前主功能

### 1. Streamlit 财务规划器

主入口是 [`app.py`](./app.py)。它做的是月度粒度的蒙特卡洛财务模拟，不是加密货币回测。

核心能力：

- 收入模拟：收入上限、职业峰值、峰值后衰减
- 职业状态机：在职、跳槽、失业、兜底四状态切换
- 支出模拟：通胀、生活方式膨胀、失业后的支出压缩、高龄支出衰减
- 债务处理：高息债、软债、赤字融资顺序
- 投资模拟：定投、常态收益、危机年份肥尾回撤
- 事件冲击：婚礼、买车、买房、个人黑天鹅
- 输出：破产风险、债务清零时间、净资产分位数、敏感性分析、CSV 导出

### 2. 历史量化研究工具链

这条线主要服务于“下载数据 -> 生成 event-driven bars -> 做标签 -> 回测/统计分析”。

- `src/data_handler/`: Binance 原始数据下载、Dollar/Volume/Tick Bar 生成
- `src/backtest/`: Backtrader 封装、实时图表、静态报告、示例策略
- `src/strategies/`: 另一套较早的向量化策略接口
- `notebooks/`: 后期主要研究工作，明显受 *Advances in Financial Machine Learning* 影响

## Notebooks 一句话地图

- `notebooks/data_process/crypto_data_download.ipynb`
  用 `RawDataDownloader` 下载 Binance 数据
- `notebooks/data_process/generate_bar.ipynb`
  用 `BarGenerator` + `BarRule` 生成 event-driven bars
- `notebooks/data_process/handle_price.ipynb`
  做 return、ADF、分数阶差分（FFD）与平稳性分析
- `notebooks/data_process/add_label.ipynb`
  做动态波动率、CUSUM filter、Triple Barrier Labeling
- `notebooks/research/analyze_dollar_bar_stats.ipynb`
  分析 dollar bar 的日内数量、时距、成交额分布
- `notebooks/research/analyze_dollar_bar_threshold.ipynb`
  比较静态阈值与动态阈值，研究 bar 数量稳定性

如果你只想知道这个仓库“后期到底在研究什么”，重点看后四个 notebook。

## 运行入口

### 财务规划器

`pyproject.toml` 目前只覆盖了量化工具链依赖，没有把 `streamlit` 和 `plotly` 写进去，所以跑 `app.py` 需要额外安装。

```bash
pip install -e .
pip install streamlit plotly
streamlit run app.py
```

可选依赖：

- `scikit-image`：仅用于敏感性分析里的等值线提取；缺失时应用仍可运行

### 旧回测示例

```bash
python script/run_backtest.py
```

但要注意：

- 它依赖一个已经整理好的 K 线 CSV
- `src/backtest/data_loader.py` 期望的是旧版列名格式
- 下载脚本/合并脚本产出的数据格式和它并不完全对齐

所以旧回测代码更适合“读结构”或“继续修”，不适合直接把它当成开箱即用产品。

## 仓库结构

```text
CryptoQuant/
├── app.py                       # 当前主入口：财务规划器
├── saved_params.json            # Streamlit 参数持久化
├── src/
│   ├── backtest/                # 历史回测框架
│   ├── data_handler/            # 数据下载与 bar 生成
│   ├── strategies/              # 较早的向量化策略接口
│   └── utils/                   # logger 等工具
├── notebooks/                   # 后期主要研究工作
├── script/                      # 独立脚本，质量和维护状态不一致
├── docs/tutorials/              # 本次整理后的精简文档
└── docs/notes/                  # 个人知识库/量化路线图，不是系统说明书
```

## 当前判断

这个仓库最真实的描述不是“一个完整统一的量化框架”，而是：

- 一个仍在使用的个人财务模拟器
- 一套保留下来的加密量化研究资产
- 一批明显已经过时的测试、脚本和旧文档

如果后面要继续维护，建议把它当成“混合仓库”处理，而不是继续假设所有模块仍然严丝合缝。
