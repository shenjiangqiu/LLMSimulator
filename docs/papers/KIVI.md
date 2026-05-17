# KIVI: 2-bit KV Cache 量化算法

> **论文**: *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*  
> **会议**: ICML 2024  
> **作者**: Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu  
> **机构**: Rice University, Texas A&M, Stevens, CMU  
> **代码**: https://github.com/jy-yuan/KIVI

---

## 一、核心问题

LLM 推理中 KV Cache 是内存和带宽瓶颈：

- **内存**：OPT-175B，batch=512，seq_len=2048 → KV Cache 3TB（3× 模型参数）
- **速度**：每生成一个 token，GPU 需从显存加载完整 KV Cache 到 SRAM，计算核心空闲

**最直接的解决方案**：量化 KV Cache，减少字节数。

---

## 二、关键发现：K 和 V 应该沿不同维度量化

### K Cache：per-channel 量化

```
K Cache 分布特征:                     量化方案:
                                      
Channel ┌─────────────────────┐      ┌─────────────────────┐
  0     │ · · · · · · · · · ·│      │ 量化组 0: channel 0  │
  1     │ · · · · · · · · · ·│      │ 量化组 1: channel 1  │
  2     │█████████████████████│ ← 异常│ 量化组 2: channel 2  │ ← 单独处理异常 channel
  3     │ · · · · · · · · · ·│      │ 量化组 3: channel 3  │
  ...   │ · · · · · · · · · ·│      │  ...                 │
        └─────────────────────┘      └─────────────────────┘
           Token →                       Token →
```

- K Cache 中某些固定 channel 的幅度非常大（outlier channel），与 AWQ/SmoothQuant 发现一致
- **per-channel 量化**将误差限定在各自的 channel 内，异常 channel 不影响正常 channel

### V Cache：per-token 量化

```
V Cache 分布特征:                     量化方案:
                                      
Token  ┌─────────────────────┐      ┌─────────────────────┐
  0    │ · · · · · · · · · ·│      │ 量化组 0: token 0    │ ← 每个 token 独立量化
  1    │ · · · · · · · · · ·│      │ 量化组 1: token 1    │
  2    │ █ █ · · · █ · · · ·│      │ 量化组 2: token 2    │
  ...  │ · · · · · · · · · ·│      │  ...                 │
       └─────────────────────┘      └─────────────────────┘
          Channel →                     Channel →
```

- V Cache 没有明显的 channel outlier 模式
- 但 attention output = Σ(attention_score × V)，是一个**加权求和**
- Attention score 高度稀疏（84.3% 接近零），只有少数 token 的 V 对输出贡献大
- **per-token 量化**保证重要 token 的 V 不受其他 token 量化误差影响

### 量化配置对比

| 配置 | CoQA (Llama-2-13B) | TruthfulQA | 错误分析 |
|------|-------------------|------------|---------|
| FP16 | 66.37 | 29.53 | baseline |
| 4bit (K=T, V=T) | 66.48 | 29.51 | 几乎无损 |
| 2bit (K=T, V=T) | 52.93 | 24.98 | 明显下降 |
| 2bit (K=C, V=C) | 2.88 | 0.74 | 崩溃 |
| 2bit (K=T, V=C) | 2.80 | 0.26 | 崩溃 |
| **2bit (K=C, V=T)** | **63.53** | **28.60** | **几乎无损！** |

> T = per-Token, C = per-Channel

---

## 三、KIVI 算法

### 核心挑战

- per-token 量化适配 streaming：新 token 直接 append
- per-channel 量化 **跨 token**，无法在 streaming 设定下直接实现

### KIVI 方案：Grouped + Residual

```
KV Cache 分为两部分:

┌────────────────────────────────┬──────────────┐
│     Grouped KV Cache           │ Residual     │
│   (量化存储, 每 G 个 token 一组) │ (FP16 保留)  │
│                                │ 最多 R 个    │
│  [Group 0][Group 1][Group 2].. │ [最新 tokens] │
└────────────────────────────────┴──────────────┘
                                   ↑
                              新 token 加入这里
                              满了 → 量化 → 移到 Grouped
```

**算法流程**（以 K Cache 为例）：

```
Prefill 阶段:
  1. 计算 K = X × W_K
  2. 将完整 K 传给下一层（保证精度）
  3. 将 K 分组: 每 G 个 token 一组
  4. 对每组 per-channel 量化 → Q(K_grouped)
  5. 不足一组的 token → K_residual (FP16 保留)

Decode 阶段:
  1. 新 K 追加到 K_residual
  2. K_residual 达到 R 个 token 时:
     a. 分组量化 → Q(K_new)
     b. 合并到 Q(K_grouped)
     c. 重置 K_residual
  3. Attention 计算:
     A = Concat(Q·Q(K_grouped)^T, Q·K_residual^T)  // tiled matmul
```

### 两个关键超参数

| 参数 | 含义 | 默认值 | 作用 |
|------|------|--------|------|
| `G` (group size) | 每组的 token 数 | 32 | 量化粒度，越大量化越高效 |
| `R` (residual length) | FP16 保留的 token 数 | 128 | **关键！**保留最近 token 的完整精度 |

### 为什么 Residual 很重要

- **Residual = FP16 滑动窗口**，保留最近 R 个 token 的完整 KV
- 对 GSM8K 等困难任务至关重要：fake 2bit 量化掉 15% 精度，KIVI 只掉 2%
- 开销可忽略：R ≤ 128，而总序列长度通常是 R 的数十倍

---

## 四、系统实现

### GPU 实现

```
量化 (Triton kernel):
  - 分组量化，每组 G 个 token 或 G 个 channel
  - 计算 min/max → scale/zero-point → 量化

反量化 + 矩阵乘 (CUDA fused kernel):
  - Q_MatMul: 在 tiling 级别融合反量和和矩阵乘
  - 避免将量化数据先展开为 FP16 再计算
  - 兼容 weight-only quantization
```

### 内存分析

**Llama-2-7B**，batch=128，seq_len=2048：

| 组件 | FP16 | KIVI-2bit | 节省 |
|------|------|-----------|------|
| 模型权重 | 12.5 GB | 12.5 GB | — |
| KV Cache | 48 GB | 6 GB | **8×** |
| 总计 | 60.5 GB | 18.5 GB | **3.3×** |

---

## 五、实验结果

### 精度（LM-Eval）

| 模型 | 配置 | CoQA | TruthfulQA | GSM8K |
|------|------|------|-----------|-------|
| Llama-2-7B | FP16 | 65.1 | 30.0 | 14.5 |
| | KIVI-2bit | 64.8 | 29.5 | 13.1 |
| Llama-2-13B | FP16 | 66.4 | 29.5 | 25.7 |
| | KIVI-2bit | 63.5 | 28.6 | 23.5 |
| Mistral-7B | FP16 | 71.1 | 42.3 | 37.5 |
| | KIVI-2bit | 70.3 | 41.6 | 36.2 |

精度损失 < 2%，在多数任务上几乎无感知。

### 长上下文（LongBench, 8192 tokens）

KIVI-2bit 在 Qasper, QMSum, MultiNews, TREC, TriviaQA 等任务上均保持与 FP16 接近的分数。

### Needle-in-a-Haystack

在 32K token 上下文中检索特定信息，KIVI-2bit 的检索精度与 FP16 baseline 几乎一致。

### 吞吐提升

| 模型 | Batch Size 提升 | 吞吐提升 |
|------|----------------|---------|
| Llama-2-7B | 4× | **2.35× ~ 3.47×** |
| 峰值内存 | ↓ 2.6× | — |

---

## 六、与 LLMSimulator / Idea 的关系

### 启示

1. **非对称量化**：K 和 V 应该沿不同维度量化，这对 PIM 实现有直接影响
   - K Cache per-channel → PIM 中每 channel 独立处理，可以并行
   - V Cache per-token → PIM 中每 token 独立，适合 streaming

2. **Residual 窗口**：FP16 保留最近 token，PIM 中可以用 GPU 处理 residual 部分

3. **解量化融合**：Q_MatMul 的 fused kernel 思路可移植到 Logic-PIM 的 GEMM module

### 在 LLMSimulator 中的建模

```
KIVI + PIM 建模要点:
  - K Cache 2-bit per-channel: PIM 从 DRAM 读 2-bit K → dequant → Q·K^T
  - V Cache 2-bit per-token: 类似
  - Residual (FP16): 保留在 GPU 或 PIM 的 buffer 中
  - G=32: 对应 PIM bundle 大小
  - 解量化开销: 查表/乘法，在 Logic die 上实现，1-2 cycles
```

---

## 七、关键术语

| 术语 | 含义 |
|------|------|
| per-token quantization | 沿 token 维度分组量化，每组包含一个 token 的所有 channel |
| per-channel quantization | 沿 channel 维度分组量化，每组包含一个 channel 的所有 token |
| group-wise quantization | 将多个 token/channel 分为一组，共享 scale/zero-point |
| residual cache | 保留 FP16 精度的最新 token，作为量化缓存的补充 |
| Q_MatMul | 融合反量化和矩阵乘的 CUDA kernel |
| NIAH | Needle-in-a-Haystack，长上下文检索任务 |
