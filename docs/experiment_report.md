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
| Asymmetric Quant | **全部启用** (z_Q=1.0, z_K=2.0) |

## 2. 5 组试验对比

| # | 试验 | KV 精度 | Q@K | Softmax | Score@V | Asym Quant |
|---|------|---------|-----|---------|---------|-----------|
| 1 | **FP16 GPU** | float16 | GPU | GPU | GPU | N/A (GPU) |
| 2 | **FP16 PIM** | float16 | PIM | GPU | PIM | ✓ |
| 3 | **2bit GPU** | 2-bit 量化 | GPU | GPU | GPU | N/A (GPU) |
| 4 | **2bit Hybrid** | 2-bit 量化 | PIM | GPU | GPU | ✓ |
| 5 | **2bit All-PIM** | 2-bit 量化 | PIM | GPU | PIM | ✓ |

> 注：所有试验中 softmax 始终在 GPU 上完成。Asymmetric quant 仅对 PIM 的 GEMV 操作生效（GPU 路径不受影响）。

### 2.1 每步延迟对比 (per decode step, 16 seqs/DP)

| Exp | Q@K (us) | per seq | Softmax (us) | Score@V (us) | AttnGen (us) | 处理器 |
|-----|---------|---------|-------------|-------------|-------------|--------|
| 1 FP16 GPU | 97 | 6.0 | ~0 | — | 95 | GPU |
| 2 FP16 PIM | 563 | 35.2 | 0.034 | 565 | 1,128 | PIM |
| 3 2bit GPU | 48 | 3.0 | ~0 | — | 48 | GPU |
| 4 2bit Hybrid | 298 | 18.6 | 0.008 | 34 | 332 | PIM |
| 5 2bit All-PIM | 298 | 18.6 | 0.008 | 299 | 596 | PIM |

> exp4 和 exp5 的 Q@K 完全一致（298us），因为两者都是 2-bit PIM + asymmetric quant。唯一区别是 Score@V 在 GPU (exp4: 34us) 还是 PIM (exp5: 299us)。

### 2.2 端到端总时间 (128 steps × 32 layers)

| Exp | 总时间 (us) | 单层时间 (us) |
|-----|------------|-------------|
| 1 FP16 GPU | 5,623 | 176 |
| 2 FP16 PIM | 38,642 | 1,208 |
| 3 2bit GPU | 2,769 | 87 |
| 4 2bit Hybrid | 11,928 | 373 |
| 5 2bit All-PIM | 20,407 | 638 |

### 2.3 Asymmetric Quant 影响

| 对比 | w/o Asym | w/ Asym | 增幅 |
|------|---------|---------|------|
| exp2 FP16 PIM Q@K | 299us | 563us | **1.88x** |
| exp4 2bit Hybrid Q@K | 166us | 298us | **1.80x** |

增幅来源：`(M+N)×K = (1+16528)×128 ≈ 2.1M` 次额外 element-sum，与主 GEMV 的数据量相当（当 M=1, N 大时）。Reduction 复用 rowbuffer 数据流，仅增加 PE 时间。

### 2.4 关键发现

1. **GPU 始终最快**：per-seq Q@K 3.0~6.0us。PIM 受限于 per-bank PE 算力（32 GOPS vs GPU 989 TFLOPS）。

2. **Hybrid (exp4) 是 PIM 方案中最优**：Q@K 在 PIM，Score@V 在 GPU。总时间 11,928us，比纯 PIM (exp2: 38,642us) 快 3.2x。

3. **exp4 vs exp5**：Q@K 相同（298us），区别仅在 Score@V（34us GPU vs 299us PIM）。GPU 做 Score@V 快 8.8x。

4. **Asymmetric quant 开销显著**：~1.8x PE 时间增加。Decode 场景下 M=1, N 大 → reduction 与主 GEMV 同规模。

5. **2-bit 优于 FP16**：PIM Q@K per-seq 18.6us (2bit) vs 35.2us (FP16)，约 1.9x。2-bit 数据量是 FP16 的 1/2（precision_byte=1 vs 2）。

## 3. PIM Cycle 验算

### 3.1 FP16 PIM (exp2, with asymmetric quant)

```
M=1, K=128, N=16528, elem=2B, 256 banks, group=4, 8 KV heads

total_bytes = (128 + 2,115,584 + 16,528) × 2 = 4,264,480 B
per_bank = 16,658 B, rb_fills = 9
rb_time = 576 ns, pe_time_main = 520.6 ns

reduction: (1+16528)×128 = 2,115,712 adds
per_bank_red = 2,115,712/256 × 2 = 16,529 B-equivalent
pe_time_red = 16,529/16 × 0.5 = 516.5 ns

total_pe = 520.6 + 516.5 = 1037.1 ns
latency = 64 + max(8×64, 1037.1) = 1101.1 ns
per_seq = 1101.1 × 4 × 8 / 1000 = 35.2 µs

Reported per seq: 563/16 = 35.2 µs
Ratio: 1.00x ✓
```

### 3.2 2bit PIM (exp4/exp5, with asymmetric quant)

```
elem=1B (2-bit)

total_bytes = 2,132,240 × 1 = 2,132,240 B
per_bank = 8,329 B, rb_fills = 5
rb_time = 320 ns, pe_time_main = 260.3 ns

reduction: 2,115,712 adds
per_bank_red = 2,115,712/256 × 1 = 8,265 B-equivalent
pe_time_red = 8,265/16 × 0.5 = 258.3 ns

total_pe = 260.3 + 258.3 = 518.6 ns
latency = 64 + max(4×64, 518.6) = 582.6 ns
per_seq = 582.6 × 4 × 8 / 1000 = 18.6 µs

Reported per seq: 298/16 = 18.6 µs
Ratio: 1.00x ✓
```

## 4. 配置说明

```yaml
nearbank_pim:
  enable_nearbank_model: true
  num_pim_banks: 256
  rowbuffer_size_bytes: 2048
  pe_width_bytes: 16.0
  pe_cycle_time_ns: 0.5
  dram_to_rb_bw_per_bank: 3.2e10
  enable_scoring_in_pim: true
  enable_context_in_pim: true
  enable_softmax_in_pim: false
  enable_asymmetric_quant: true     # 全部试验启用
  zero_point_q: 1.0
  zero_point_k: 2.0
```

| 参数 | exp1 | exp2 | exp3 | exp4 | exp5 |
|------|------|------|------|------|------|
| processor_type | GPU | GPU+PIM | GPU | GPU+PIM | GPU+PIM |
| precision_byte | 2 | 2 | 1 | 1 | 1 |
| enable_scoring_in_pim | — | true | — | true | true |
| enable_context_in_pim | — | true | — | false | true |
| enable_asymmetric_quant | true | true | true | true | true |
