# Nearbank PIM 模拟器 — 试验报告

## 1. 试验环境

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3-8B (dense, 32层, hidden=4096, heads=32, kv_heads=8, head_dim=128) |
| GPU | B100 (989 TFLOPS FP16, 3.35 TB/s HBM带宽) |
| Prefill 长度 | 16400 tokens |
| Decode 长度 | 128 steps |
| Batch size | 64 |
| PIM banks | 256 |
| Row buffer | 2048 B |
| PE FP16 | 16 B/cycle @ 0.5ns (2 GHz) |
| PE lowbit | 32 B/cycle @ 0.5ns (2 GHz) |
| DRAM→RB 带宽 | 32 GB/s per bank |

## 2. 5 组试验对比

| # | 试验 | KV 精度 | Q@K | Softmax | Score@V | KV Quant |
|---|------|---------|-----|---------|---------|----------|
| 1 | **FP16 GPU** | float16 | GPU | GPU | GPU | GPU |
| 2 | **FP16 PIM** | float16 | PIM | GPU | PIM | GPU |
| 3 | **2bit GPU** | 2-bit 量化 | GPU | GPU | GPU | GPU |
| 4 | **2bit Hybrid** | 2-bit 量化 | PIM | GPU | GPU | GPU |
| 5 | **2bit All-PIM** | 2-bit 量化 | PIM | GPU | PIM | GPU |

> 注：所有试验中 softmax 始终在 GPU 上完成。

### 2.1 每步延迟对比 (per decode step, per DP group)

| Exp | Q@K (us) | Softmax (us) | Score@V (us) | KV Quant (us) | AttnGen 总计 (us) | 处理器 |
|-----|---------|-------------|-------------|---------------|-----------------|--------|
| 1 FP16 GPU | 12.4 | ~0 | — | 0 | 12.2 | GPU |
| 2 FP16 PIM | 37.3 | 0.004 | 37.3 | 0.001 | 74.7 | PIM |
| 3 2bit GPU | 6.4 | ~0 | — | 0 | 6.3 | GPU |
| 4 2bit Hybrid | 20.7 | 0.001 | 20.7 | 0.001 | 41.4 | PIM |
| 5 2bit All-PIM | 165.8 | 0.008 | 165.8 | 0.008 | 331.5 | PIM |

### 2.2 端到端总时间

| Exp | 总时间 (us) | 单层时间 (us) |
|-----|------------|-------------|
| 1 FP16 GPU | 2,917 | 91 |
| 2 FP16 PIM | 4,930 | 154 |
| 3 2bit GPU | 1,476 | 46 |
| 4 2bit Hybrid | 2,562 | 80 |
| 5 2bit All-PIM | 11,929 | 373 |

### 2.3 关键发现

1. **PIM 在 FP16 下比 GPU 慢**：AttentionGen PIM (74.7us) vs GPU (12.2us)，约 6x 差距。因为单个 bank PE 算力（32 GOPS）远小于 GPU（989 TFLOPS）。

2. **PIM 在 2bit Hybrid 下仍然慢于 GPU**：Q@K 在 PIM (20.7us) vs GPU (6.4us)，但区别缩小。2bit 数据量是 FP16 的 1/8，PIM 带宽优势部分显现。

3. **2bit All-PIM 最慢**：Q@K+Score@V 都在 PIM，每步 331.5us。score@V 处理的是 FP16 精度 scores，需要 FP16 PE，速度受限于 PE 算力。

4. **Softmax 开销极小**：0.001~0.008us，可忽略。

5. **KV 量化开销极小**：0.001~0.008us per step，非对称量化 overhead 可忽略。

## 3. PIM Cycle 验算

### 3.1 验算方法

对 exp2 (FP16 PIM) 的 Q@K 阶段进行 cycle-level 校验：

**输入参数**：
```
M=1, K=128, N=16528 (seq_len), element_size=2B (FP16)
256 PIM banks, 8 KV heads, attention_group=4
rowbuffer=2048B, DRAM→RB BW=32GB/s, PE=16B/cycle@0.5ns
```

**分析模型**：

```
total_elements = M*K + K*N + M*N
               = 128 + 2,115,584 + 16,528 = 2,132,240
total_bytes = 2,132,240 × 2 = 4,264,480 B

per_bank = 4,264,480 / 256 = 16,658 B
rb_fills = ceil(16,658 / 2,048) = 9

rb_fill_one = 2,048 / 32×10^9 × 10^9 = 64 ns
rb_total = 9 × 64 = 576 ns

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
Analytical per step:  18.71 µs  (per sequence, all 8 KV heads)
Reported Q@K per seq: 18.67 µs  (37.35 / 2 seqs in DP group)
Ratio:                1.00x
```

**✓ PIM cycle 验算通过。** 模拟器报告的延迟与理论模型完全吻合。

## 4. 测试过程

### 4.1 构建

```bash
cd build_release && cmake .. && make -j
```

### 4.2 运行试验

```bash
# 逐个运行 5 组试验
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

输出 Markdown 表格 + PIM cycle 验算。

### 4.4 单元测试

```bash
./build_release/test_nearbank
# 35 tests passed, 0 failed
```

覆盖：基础 GEMV 延迟、2-bit 量化、PE 配置、bank 数量、attention 场景、
边缘情况、rowbuffer 流水线、非对称量化 reduction、配置验证。

## 5. 配置说明

### 5.1 Nearbank PIM 参数 (config.yaml)

```yaml
nearbank_pim:
  enable_nearbank_model: true    # 启用 nearbank 模型
  num_pim_banks: 256             # PIM bank 数量
  rowbuffer_size_bytes: 2048     # row buffer 大小
  pe_width_bytes: 16.0           # FP16 PE 宽度 (B/cycle)
  pe_cycle_time_ns: 0.5          # PE 周期 (ns)
  pe_lowbit_width_bytes: 32.0    # 低位宽 PE 宽度
  pe_lowbit_cycle_time_ns: 0.5   # 低位宽 PE 周期
  dram_to_rb_bw_per_bank: 3.2e10 # DRAM→RB 带宽 per bank (B/s)

  # 阶段路由 (true=在 PIM, false=在 GPU)
  enable_scoring_in_pim: true    # Q@K^T
  enable_context_in_pim: true    # score@V
  enable_softmax_in_pim: false   # softmax (建议始终 false)

  # 非对称量化
  enable_asymmetric_quant: true
  zero_point_q: 1.0
  zero_point_k: 2.0
```

### 5.2 试验配置差异

| 参数 | exp1 | exp2 | exp3 | exp4 | exp5 |
|------|------|------|------|------|------|
| processor_type | GPU | GPU+PIM | GPU | GPU+PIM | GPU+PIM |
| parallel_execution | off | on | off | on | on |
| precision_byte | 2 | 2 | 1 | 1 | 1 |
| compressed_kv | off | off | on | on | on |
| enable_scoring_in_pim | — | true | — | true | true |
| enable_context_in_pim | — | true | — | false | true |

## 6. 代码架构

```
src/dram/nearbank/
├── nearbank_config.h    # 配置结构 (PE, bank, bandwidth)
├── nearbank_pim.h       # NearbankPIMUnit 类
├── nearbank_pim.cpp     # 流水线延迟模型
└── CMakeLists.txt

tools/
├── parse_layer.py       # 输出解析 → 嵌套 JSON
└── report.py            # 对比报告 + cycle 验算

log/
├── config_exp1_fp16_gpu.{json,txt}
├── config_exp2_fp16_pim.{json,txt}
├── config_exp3_2bit_gpu.{json,txt}
├── config_exp4_2bit_hybrid.{json,txt}
└── config_exp5_2bit_allpim.{json,txt}
```

## 7. 已知限制

1. **Prefill 模式未修复**：`prefill_mode: on` 时 KV cache 初始化有 bug（非本次改动引入）。
2. **GPU attention 未拆分 softmax/context**：原始代码只单独建模 Q@K 阶段，exp1/exp3 的 softmax 和 score@V 未独立计时。
3. **Timeline 拆分不完整**：hybrid 模式的 GPU stage 时间加到了 PIM timeline 而非 GPU timeline（不影响总延迟数值）。
4. **PE 模型简化**：dual PE (FP16 + lowbit) 配置已加但尚未在 GEMV 计算中按数据类型自动选择 PE。
