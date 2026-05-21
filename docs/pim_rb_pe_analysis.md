# PIM Rowbuffer & PE 时间分析

## 1. 配置参数

| 参数 | 值 |
|------|-----|
| PIM banks | 2048 |
| Row buffer | 2048 B |
| PE FP16 | 16 B/cycle @ 0.5ns (2GHz) |
| DRAM→RB 带宽 | 32 GB/s per bank |
| 模型 | Llama-3-8B: head_dim=128, kv_heads=8, group=4 |
| Sequence | M=1, N=16528 (seq_len), 16 seqs/DP |

## 2. 数据量拆解

GEMV 操作：$Q[M \times K] \cdot K^T[K \times N] \rightarrow S[M \times N]$

总元素数 = 读 A + 读 B + 写结果：

$$E_{\text{total}} = M \cdot K + K \cdot N + M \cdot N$$

### FP16（elem=2B）

$$E_{\text{total}} = 128 + 128 \times 16528 + 16528 = 2,132,240 \text{ elements}$$

$$B_{\text{total}} = 2,132,240 \times 2 = 4,264,480 \text{ B}$$

$$B_{\text{bank}} = 4,264,480 / 2048 = 2,082 \text{ B}$$

$$n_{\text{rb}} = \lceil 2082 / 2048 \rceil = 2$$

### 2-bit（elem=1B，precision_byte=1）

$$B_{\text{total}} = 2,132,240 \times 1 = 2,132,240 \text{ B}$$

$$B_{\text{bank}} = 2,132,240 / 2048 = 1,041 \text{ B}$$

$$n_{\text{rb}} = \lceil 1041 / 2048 \rceil = 1$$

## 3. Rowbuffer 时间

每个 rowbuffer fill 时间固定：

$$t_{\text{fill}} = \frac{2048 \text{ B}}{32 \times 10^9 \text{ B/s}} = 64 \text{ ns}$$

| 精度 | per_bank | rb_fills | rb 总时间 |
|------|----------|----------|----------|
| FP16 | 2,082 B | 2 | $2 \times 64 = 128$ ns |
| 2-bit | 1,041 B | 1 | $1 \times 64 = 64$ ns |

**FP16 的 rb 时间是 2-bit 的 2 倍**（128ns vs 64ns），因为数据量翻倍导致需要 2 次 rowbuffer fill。

## 4. PE 计算时间

PE 每周期处理 $W = 16$ 字节。每个 GEMV 的 PE 时间：

$$T_{\text{pe}} = \frac{B_{\text{bank}}}{W} \cdot \tau$$

| 精度 | $B_{\text{bank}}$ | PE 时间 |
|------|-------------------|---------|
| FP16 | 2,082 B | $2082/16 \times 0.5 = 65.1$ ns |
| 2-bit | 1,041 B | $1041/16 \times 0.5 = 32.5$ ns |

**FP16 的 PE 时间是 2-bit 的 2 倍**（65.1ns vs 32.5ns），与数据量成正比。

## 5. 非对称量化（Asymmetric Quant）的额外开销

当启用非对称量化时，额外增加 reduction 操作：

$$\text{reduction\_elements} = (M + N) \times K = (1 + 16528) \times 128 = 2,115,712$$

$$T_{\text{red}} = \frac{2,115,712 \times e}{2048 \times 16} \times 0.5$$

| 精度 | $T_{\text{red}}$ | 说明 |
|------|-----------------|------|
| FP16 (e=2) | $2,115,712 \times 2 / 2048 / 16 \times 0.5 = 64.6$ ns | 当前试验未启用 |
| 2-bit (e=1) | $2,115,712 \times 1 / 2048 / 16 \times 0.5 = 32.3$ ns | exp4/5/6 启用 |

## 6. 流水线延迟模型

$$T_{\text{latency}} = t_{\text{fill}} + \max((n_{\text{rb}}-1) \cdot t_{\text{fill}},\; T_{\text{pe}} + T_{\text{red}})$$

### FP16 PIM（exp2，无 asymmetric quant）

$$T_{\text{latency}} = 64 + \max(1 \times 64,\; 65.1 + 0) = 64 + 65.1 = 129.1 \text{ ns}$$

### 2-bit PIM（exp5，有 asymmetric quant）

$$T_{\text{latency}} = 64 + \max(0 \times 64,\; 32.5 + 32.3) = 64 + 64.8 = 128.8 \text{ ns}$$

**关键发现：FP16 和 2-bit+asym 的 per-GEMV 延迟几乎相同**（129ns vs 129ns），因为：
- FP16 数据量大 2× → rb/pe 时间长 2×
- 但 2-bit 的 asymmetric quant 额外 reduction 把 PE 时间也翻倍了
- 两者恰好抵消

## 7. Per-Step 汇总

每个 decode step 包含两个 GEMV 方向（Q@K + Score@V），每个方向跨越 8 个 KV heads × 4 个 Q heads per KV head = 32 个 attention group：

$$T_{\text{per\_direction}} = T_{\text{latency}} \times 32 \times 16 \text{ seqs}$$

| 精度 | per-GEMV | per-direction | qk_rb | qk_pe | sv_rb | sv_pe |
|------|----------|--------------|-------|-------|-------|-------|
| FP16 | 129.1ns | 66.1us | 65.5us | 33.3us | 65.5us | 33.3us |
| 2-bit | 128.8ns | 66.0us | 32.8us | 33.2us | 32.8us | 33.2us |

### 验算：

**FP16 rb per direction**：
$$128\text{ns} \times 32 \times 16 = 65,536\text{ns} = 65.5\mu s \quad \checkmark$$

**FP16 pe per direction**：
$$65.1\text{ns} \times 32 \times 16 = 33,331\text{ns} = 33.3\mu s \quad \checkmark$$

**2-bit rb per direction**：
$$64\text{ns} \times 32 \times 16 = 32,768\text{ns} = 32.8\mu s \quad \checkmark$$

**2-bit pe per direction（含 asymmetric）**：
$$(32.5 + 32.3)\text{ns} \times 32 \times 16 = 33,178\text{ns} = 33.2\mu s \quad \checkmark$$

## 8. 总结

| 因素 | FP16 PIM | 2-bit PIM | 比值 |
|------|----------|-----------|------|
| 每元素字节 | 2 B | 1 B | 2:1 |
| per_bank 数据量 | 2,082 B | 1,041 B | 2:1 |
| rb_fills | 2 | 1 | 2:1 |
| 纯 PE 时间 | 65.1 ns | 32.5 ns | 2:1 |
| Asym reduction | — | 32.3 ns | — |
| 总 PE 时间 | 65.1 ns | 64.8 ns | **≈1:1** |
| per-GEMV 延迟 | 129.1 ns | 128.8 ns | **≈1:1** |

核心 tradeoff：**2-bit 省了带宽（rb 从 2 fills→1 fill），但 asymmetric quant 的 reduction 把 PE 时间加回来了。最终 per-GEMV 延迟几乎相同，但 2-bit 的 KV cache 占用只有 FP16 的 1/8。**
