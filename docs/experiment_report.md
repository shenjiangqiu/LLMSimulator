# Nearbank PIM 模拟器 — 试验报告

## 1. 试验环境

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3-8B (dense, 32层, hidden=4096, heads=32, kv_heads=8, head_dim=128) |
| GPU | B100 (989 TFLOPS FP16, 3.35 TB/s HBM带宽) |
| Prefill 长度 | 16400 tokens |
| Decode 长度 | 128 steps |
| Batch size | 64 (16 seqs/DP group) |
| PIM banks | 256 |
| Row buffer | 2048 B |
| PE FP16 | 16 B/cycle @ 0.5ns (2 GHz) |
| PE lowbit | 32 B/cycle @ 0.5ns (2 GHz) |
| DRAM→RB 带宽 | 32 GB/s per bank |

## 2. 5 组试验对比

| # | 试验 | KV 精度 | Q@K | Softmax | Score@V | Asym Quant |
|---|------|---------|-----|---------|---------|-----------|
| 1 | **FP16 GPU** | float16 | GPU | GPU | GPU | — |
| 2 | **FP16 PIM** | float16 | PIM | GPU | PIM | — |
| 3 | **2bit GPU** | 2-bit 量化 | GPU | GPU | GPU | ✓ |
| 4 | **2bit Hybrid** | 2-bit 量化 | PIM | GPU | GPU | ✓ |
| 5 | **2bit All-PIM** | 2-bit 量化 | PIM | GPU | PIM | ✓ |

> 注：Asymmetric quant 仅对 2-bit KV 的 PIM GEMV 操作生效。FP16 无需量化，不产生额外 reduction。

### 2.1 每步延迟对比 (per decode step, 16 seqs/DP)

| Exp | Q@K (us) | per seq | Softmax (us) | Score@V (us) | AttnGen (us) | 处理器 |
|-----|---------|---------|-------------|-------------|-------------|--------|
| 1 FP16 GPU | 97 | 6.0 | ~0 | — | 95 | GPU |
| 2 FP16 PIM | 299 | 18.7 | 0.034 | 299 | 598 | PIM |
| 3 2bit GPU | 48 | 3.0 | ~0 | — | 48 | GPU |
| 4 2bit Hybrid | 298 | 18.6 | 0.008 | 34 | 332 | PIM |
| 5 2bit All-PIM | 298 | 18.6 | 0.008 | 299 | 596 | PIM |

> exp2 (FP16 PIM) 和 exp4/5 (2bit PIM) 的 per-seq Q@K 几乎相同（18.7 vs 18.6us）。
> FP16 数据量大 2x → rowbuffer/PE 时间长 → 但 2-bit 有 asymmetric quant 的额外 reduction → 两者恰好抵消。

### 2.2 端到端总时间

| Exp | 总时间 (us) | 单层时间 (us) |
|-----|------------|-------------|
| 1 FP16 GPU | 5,623 | 176 |
| 2 FP16 PIM | 21,684 | 678 |
| 3 2bit GPU | 2,769 | 87 |
| 4 2bit Hybrid | 11,928 | 373 |
| 5 2bit All-PIM | 20,407 | 638 |

### 2.3 Asymmetric Quant 影响（2-bit PIM 场景）

| 对比 | w/o Asym | w/ Asym | 增幅 |
|------|---------|---------|------|
| exp4 Q@K | 166us | 298us | **1.80x** |

增幅来源：`(M+N)×K = (1+16528)×128 ≈ 2.1M` 次额外 element-sum。Decode 场景 M=1 时，reduction 与主 GEMV 数据量相当，PE 时间翻倍。

### 2.4 关键发现

1. **GPU 始终最快**：per-seq Q@K 3.0~6.0us。PIM 受限于 per-bank PE 算力。

2. **Hybrid (exp4) 是 PIM 方案中最优**：Q@K 在 PIM，Score@V 在 GPU。总时间 11,928us。

3. **exp4 vs exp5**：Q@K 相同（298us），Score@V 差异巨大（34us GPU vs 299us PIM）——GPU 做 Score@V 快 8.8x。

4. **Asymmetric quant 开销显著**：~1.8x PE 时间。但与 FP16 的数据量优势（2x）恰好对冲，导致 2-bit PIM 和 FP16 PIM 的 per-seq Q@K 接近。

5. **FP16 PIM vs 2bit PIM**：per-seq Q@K 几乎相同（18.7 vs 18.6us）。2-bit 的带宽优势被 asymmetric quant 的 PE 开销抵消。

## 3. PIM Cycle 验算

### 3.1 FP16 PIM (exp2, no asymmetric quant)

```
M=1, K=128, N=16528, elem=2B, 256 banks, group=4

total_bytes = (128 + 2,115,584 + 16,528) × 2 = 4,264,480 B
per_bank = 16,658 B, rb_fills = 9
rb_time = 576 ns, pe_time = 520.6 ns
latency = 64 + max(8×64, 520.6) = 584.6 ns
per_seq = 584.6 × 4 × 8 / 1000 = 18.7 µs

Reported: 299/16 = 18.7 µs  →  Ratio: 1.00x ✓
```

### 3.2 2bit PIM (exp4/exp5, with asymmetric quant)

```
elem=1B

total_bytes = 2,132,240 B, per_bank = 8,329 B, rb_fills = 5
rb_time = 320 ns, pe_time_main = 260.3 ns

reduction: (1+16528)×128 = 2,115,712 adds
per_bank_red = 2,115,712/256 × 1 = 8,265 B-equivalent
pe_time_red = 258.3 ns

total_pe = 260.3 + 258.3 = 518.6 ns
latency = 64 + max(4×64, 518.6) = 582.6 ns
per_seq = 582.6 × 4 × 8 / 1000 = 18.6 µs

Reported: 298/16 = 18.6 µs  →  Ratio: 1.00x ✓
```

## 4. 配置

| 参数 | exp1 | exp2 | exp3 | exp4 | exp5 |
|------|------|------|------|------|------|
| processor_type | GPU | GPU+PIM | GPU | GPU+PIM | GPU+PIM |
| precision_byte | 2 | 2 | 1 | 1 | 1 |
| enable_scoring_in_pim | — | true | — | true | true |
| enable_context_in_pim | — | true | — | false | true |
| enable_asymmetric_quant | — | — | ✓ | ✓ | ✓ |
