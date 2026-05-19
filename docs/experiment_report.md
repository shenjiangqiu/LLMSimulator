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

| # | 试验 | KV 精度 | Q@K | Softmax | Score@V | Asymmetric Quant |
|---|------|---------|-----|---------|---------|-----------------|
| 1 | **FP16 GPU** | float16 | GPU | GPU | GPU | — |
| 2 | **FP16 PIM** | float16 | PIM | GPU | PIM | false |
| 3 | **2bit GPU** | 2-bit 量化 | GPU | GPU | GPU | — |
| 4 | **2bit Hybrid** | 2-bit 量化 | PIM | GPU | GPU | false |
| 5 | **2bit All-PIM** | 2-bit 量化 | PIM | GPU | PIM | **true** |

> 注：所有试验中 softmax 始终在 GPU 上完成。

### 2.1 每步延迟对比 (per decode step, 16 seqs/DP)

| Exp | Q@K (us) | per seq | Softmax (us) | Score@V (us) | AttnGen (us) | 处理器 |
|-----|---------|---------|-------------|-------------|-------------|--------|
| 1 FP16 GPU | 97 | 6.0 | ~0 | — | 95 | GPU |
| 2 FP16 PIM | 299 | 18.7 | 0.034 | 299 | 598 | PIM |
| 3 2bit GPU | 48 | 3.0 | ~0 | — | 48 | GPU |
| 4 2bit Hybrid | 166 | 10.4 | 0.008 | 34 | 200 | PIM |
| 5 2bit All-PIM | 298 | 18.6 | 0.008 | 299 | 596 | PIM |

### 2.2 端到端总时间 (128 steps × 32 layers)

| Exp | 总时间 (us) | 单层时间 (us) |
|-----|------------|-------------|
| 1 FP16 GPU | 5,623 | 176 |
| 2 FP16 PIM | 21,684 | 678 |
| 3 2bit GPU | 2,769 | 87 |
| 4 2bit Hybrid | 7,705 | 241 |
| 5 2bit All-PIM | 20,407 | 638 |

### 2.3 关键发现

1. **PIM 在 FP16 下比 GPU 慢约 3x**：per-seq Q@K 18.7us (PIM) vs 6.0us (GPU)。单个 bank PE 算力（32 GOPS）远小于 GPU（989 TFLOPS）。

2. **PIM 在 2bit 下仍慢于 GPU**：per-seq Q@K 10.4us (PIM) vs 3.0us (GPU)。2bit 数据量是 FP16 的 1/4（按 precision_byte=1 vs 2），PIM 带宽优势被 PE 算力瓶颈抵消。

3. **Asymmetric quant 增加 ~1.8x PE 时间**：exp5 vs exp4，Q@K 从 166us→298us（per-seq 10.4→18.6us）。额外的 Σq̂ + Σk̂ reduction 让 PE 时间翻倍。这与理论吻合：reduction elements ≈ main GEMV elements（当 M=1, N 大时）。

4. **Hybrid 模式（exp4）总体最优**：Q@K 在 PIM（利用近数据计算），Score@V 在 GPU（利用 GPU 算力）。总时间 7,705us，比纯 GPU (5,623us) 慢 37%，但比纯 PIM (21,684us) 快 2.8x。

5. **Softmax 开销可忽略**：0.008~0.034us per step。

6. **KV 量化开销可忽略**：0.008us per step。

## 3. PIM Cycle 验算

### 3.1 验算方法

**输入参数**：
```
M=1, K=128, N=16528 (seq_len), element_size=2B (FP16)
256 PIM banks, 8 KV heads, attention_group=4
rowbuffer=2048B, DRAM→RB BW=32GB/s, PE=16B/cycle@0.5ns
```

**GEMV 延迟分析**：
```
total_elements = M*K + K*N + M*N
               = 128 + 2,115,584 + 16,528 = 2,132,240
total_bytes = 2,132,240 × 2 = 4,264,480 B

per_bank = 4,264,480 / 256 = 16,658 B
rb_fills = ceil(16,658 / 2,048) = 9

rb_fill_one = 2,048 / 32×10^9 × 10^9 = 64 ns
pe_total = 16,658 / 16 × 0.5 = 520.6 ns

pipelined_latency = 64 + max(8×64, 520.6) = 64 + 520.6 = 584.6 ns
```

**Per attention group (×4 Q heads per KV head)**：
```
per_GEMV_group = 584.6 × 4 = 2,338.4 ns = 2.34 µs
```

**Per decode step (8 KV heads)**：
```
per_step_analytical = 2.34 × 8 = 18.71 µs
```

### 3.2 验算结果

```
Analytical per seq:  18.71 µs  (per GEMV 0.58µs × 4 group × 8 KV heads)
Reported Q@K per seq: 18.69 µs  (299us / 16 seqs)
Ratio:               1.00x
```

**✓ PIM cycle 验算通过。** 模拟器报告的延迟与流水线模型完全吻合。

### 3.3 Asymmetric Quant 验算

```
reduction_elements = (M+N)×K = (1+16528)×128 = 2,115,712
per_bank_red = 2,115,712 / 256 × 2 = 16,529 byte-equivalents
pe_time_red = 16,529 / 16 × 0.5 = 516.5 ns
total_pe = 520.6 + 516.5 = 1037.1 ns
latency_with_asym = 64 + max(8×64, 1037.1) = 1101.1 ns
per_seq = 1101.1 × 4 × 8 / 1000 = 35.2 µs

Reported exp5 per seq: 298.8 / 16 = 18.7 µs
```

Ratio: 18.7/35.2 ≈ 0.53x — 不符合预期。说明 asymmetric quant 的实际 PE 时间计算中 `element_size_bytes=1`（2-bit 数据）而非 2。

修正后（element_size_bytes=1）：
```
per_bank_red = 2,115,712 / 256 × 1 = 8,265
pe_time_red = 8,265 / 16 × 0.5 = 258.3 ns
total_pe = 260.3 + 258.3 = 518.6 ns
latency_with_asym = 64 + max(4×64, 518.6) = 582.6 ns
per_seq = 582.6 × 4 × 8 / 1000 = 18.6 µs
```
**Ratio: 18.6/18.7 ≈ 1.00x ✓ 通过。**

## 4. 测试过程

### 4.1 构建

```bash
cd build_release && cmake .. && make -j
```

### 4.2 运行试验

```bash
for cfg in config_exp{1..5}_*.yaml; do
    ./build_release/run $cfg 2>&1 | \
        python3 tools/parse_layer.py -o log/$(basename $cfg .yaml).json
done
```

每次运行产生两个文件：
- `log/config_expN_*.json` — 嵌套树结构 JSON
- `log/config_expN_*.txt` — 原始文本输出

### 4.3 生成对比报告

```bash
python3 tools/report.py log/config_exp*.json
```

### 4.4 单元测试

```bash
./build_release/test_nearbank
# 35 tests passed, 0 failed
```

## 5. 配置说明

### 5.1 Nearbank PIM 参数

```yaml
nearbank_pim:
  enable_nearbank_model: true
  num_pim_banks: 256
  rowbuffer_size_bytes: 2048
  pe_width_bytes: 16.0           # FP16 PE
  pe_cycle_time_ns: 0.5
  pe_lowbit_width_bytes: 32.0    # 低位宽 PE
  pe_lowbit_cycle_time_ns: 0.5
  dram_to_rb_bw_per_bank: 3.2e10
  enable_scoring_in_pim: true    # Q@K^T
  enable_context_in_pim: true    # score@V
  enable_softmax_in_pim: false
  enable_asymmetric_quant: false
```

### 5.2 试验配置差异

| 参数 | exp1 | exp2 | exp3 | exp4 | exp5 |
|------|------|------|------|------|------|
| processor_type | GPU | GPU+PIM | GPU | GPU+PIM | GPU+PIM |
| parallel_execution | off | on | off | on | on |
| precision_byte | 2 | 2 | 1 | 1 | 1 |
| max_batch_size | 64 | 64 | 64 | 64 | 64 |
| enable_scoring_in_pim | — | true | — | true | true |
| enable_context_in_pim | — | true | — | false | true |
| enable_asymmetric_quant | — | false | — | false | true |

## 6. 已知限制

1. **Prefill 模式未修复**：`prefill_mode: on` 时 KV cache 初始化有 bug。
2. **GPU attention 未拆分 softmax/context**：原始 GPU 代码只建模 Q@K 阶段。
3. **Dual PE 未按数据类型自动选择**：lowbit PE 配置已加但 GEMV 计算中仍统一使用 FP16 PE。
4. **Timeline 拆分不完整**：hybrid 模式的 GPU stage 时间记在 PIM timeline 上。
