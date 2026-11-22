It only uploads part of the data files. For getting all dataset of the CWRU, directly visit link to download it via https://engineering.case.edu/bearingdatacenter/download-data-file
```markdown
# 基于 CWRU 轴承数据集的跨工况故障诊断（0HP → 1HP）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CWRU%2012kHz%20DE-green.svg)](https://engineering.case.edu/bearingdatacenter)

## 📌 项目简介

本项目基于**凯斯西储大学（Case Western Reserve University, CWRU）**滚动轴承故障数据集，构建了一套端到端（End-to-End）一维时域振动信号深度学习故障诊断框架[cite: 1]。

项目重点聚焦**跨负载工况（Cross-Condition）泛化能力验证**：仅使用 **0 HP（空载，1797 rpm）** 工况下的数据训练与验证模型，直接在未见过的 **1 HP（1 马力负载，1772 rpm）** 工况下测试模型对**正常状态、滚动体故障、内圈故障、外圈故障**四类状态的识别性能[cite: 1]。

---

## 📂 仓库结构

```text
├── archive.zip                                             # CWRU 原始 .mat 驱动端振动信号数据压缩包
├── 02-cwru_cross_condition_preprocessing_0hp_to_1hp.ipynb  # 数据预处理：信号提取、重叠滑窗切片与 .npz 打包
├── 04-cwru_cross_condition_cnn_0hp_to_1hp.ipynb            # 1D-CNN 故障诊断模型定义、训练与跨工况评估
├── 06-cwru_cross_condition_lstm_0hp_to_1hp_完整版.ipynb     # LSTM 故障诊断模型定义、训练与跨工况评估（对比基线）
└── README.md                                               # 项目说明文档

```

---

## 📊 数据集与实验设定

1. **原始信号来源**：采用 CWRU 12 kHz 采样率下的**驱动端加速度传感器振动信号（`_DE_time`）**，涵盖 `0.007"`、`0.014"`、`0.021"`、`0.028"` 故障直径。


2. **故障类别（4 分类）**：


* `0`：正常状态（Normal）


* `1`：滚动体故障（Ball Fault）


* `2`：内圈故障（Inner Race Fault）


* `3`：外圈故障（Outer Race Fault）




3. **滑动窗口切片**：


* **窗口长度（Window Size）**：`1024`


* **滑动步长（Stride）**：`512`（50% 重叠采样）




4. **跨工况数据集划分与防泄漏标准化**：


* **训练集（0 HP）**：`2,084` 个样本（按 8:2 分层划分）


* **验证集（0 HP）**：`521` 个样本


* **跨工况测试集（1 HP）**：`3,074` 个样本（完全独立于训练过程）


* **Z-score 标准化**：严格仅基于 **0 HP 训练集**计算均值 $\mu$ 与标准差 $\sigma$，并将其应用至验证集与 1 HP 测试集，杜绝数据泄漏。





---

## 🧠 模型架构

### 1. 1D-CNN 诊断模型 (`CNNClassifier`)

直接输入形状为 `(Batch, 1, 1024)` 的一维振动信号切片，无需人工提取频域/统计特征：

* **特征提取模块 (`features`)**：


* `Conv1d(1 → 16, kernel_size=7, padding=3)` + `ReLU` + `MaxPool1d(2)`


* `Conv1d(16 → 32, kernel_size=5, padding=2)` + `ReLU` + `MaxPool1d(2)`


* `Conv1d(32 → 64, kernel_size=3, padding=1)` + `ReLU` + `AdaptiveAvgPool1d(1)`




* **分类头模块 (`classifier`)**：


* `Flatten` → `Dropout` → `Linear(64, 4)`




* **训练配置**：交叉熵损失函数（`CrossEntropyLoss`）、`Adam` 优化器（含权重衰减）及验证集早停机制（`Patience = 10`）。



### 2. LSTM 对比模型 (`06-...-lstm...ipynb`)

* 采用 2 层堆叠 `nn.LSTM`（`hidden_size = 64`）捕捉长时序周期依赖，取序列末端时间步隐藏状态接入全连接层完成 4 分类。



---

## 📈 跨工况实验结果（0 HP → 1 HP）

在 **1 HP 跨工况测试集（共 3,074 条样本）** 上的 1D-CNN 详细评估结果如下：

| 类别 / 指标 (Class / Metric) | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| **正常 (Normal)** | 1.000000 | 1.000000 | 1.000000 | 944 |
| **滚动体故障 (Ball Fault)** | 0.963315 | 1.000000 | 0.981315 | 709 |
| **内圈故障 (Inner Race Fault)** | 1.000000 | 0.902680 | 0.948851 | 709 |
| **外圈故障 (Outer Race Fault)** | 0.944297 | 1.000000 | 0.971351 | 712 |
| **Accuracy（总体准确率）** | — | — | **0.977554** | **3074** |
| **Macro Avg（宏平均）** | 0.976903 | 0.975670 | 0.975379 | 3074 |
| **Weighted Avg（加权平均）** | 0.978637 | 0.977554 | 0.977257 | 3074 |

* **结果分析**：模型在未见过的 1 HP 跨负载工况下取得了 **97.76%** 的总体识别准确率，正常、滚动体与外圈故障召回率均达到 **100%**；主要误差集中在内圈故障的跨负载特征偏移上（召回率 `90.27%`），这也是后续引入域泛化与注意力机制的主要改进方向。



---

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install numpy scipy scikit-learn torch matplotlib seaborn jupyter

```

### 2. 运行步骤

1. **解压数据**：将 `archive.zip` 解压至当前目录对应的数据路径下。


2. **数据预处理**：依次运行 `02-cwru_cross_condition_preprocessing_0hp_to_1hp.ipynb`，生成标准化后的 `.npz` 切片数据集。


3. **模型训练与评估**：运行 `04-cwru_cross_condition_cnn_0hp_to_1hp.ipynb`（或 `06-cwru_cross_condition_lstm_0hp_to_1hp_完整版.ipynb`），即可自动完成训练并输出训练曲线、混淆矩阵及分类报告。



---

## 🔮 后续研究规划

* [ ] 拓展多负载组合（`0/1/2/3 HP`）及引入 **Paderborn (PU)** 真实自然损伤轴承数据集。


* [ ] 加入不同信噪比（SNR）高斯白噪声测试模型鲁棒性。


* [ ] 引入二次卷积（QCNN）、局部注意力机制及单源域泛化（SDG）算法，进一步提升跨工况下内圈故障的召回率与模型可解释性。



```

```
