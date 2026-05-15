# Idea: 量化 KV Cache + Near-Bank PIM 加速 LLM Attention

## 核心观点

### 1. 使用量化 KV Cache（2-bit）进行计算

- LLM 推理的 memory bottleneck 主要来自 KV Cache
- 将 KV Cache 从 FP16 量化到 2-bit，内存占用减少 **8×**
- 在 PIM 中使用 2-bit 数据进行 attention 计算（Q·K^T），利用 PIM 的高带宽
- 挑战：2-bit 量化带来的精度损失需要在 attention score 层面补偿

### 2. 使用 Near-Bank（Bank-PIM / Logic-PIM）进行计算

- **Q·K^T（score 计算）**：低 Op/B（GEMV），适合 PIM
  - 从 DRAM bank 读取 2-bit K cache → 在 bank 附近做 Q·K 内积
  - 避免将大量 KV cache 数据搬移到 GPU
- **score·V（attention 输出）**：Op/B 取决于 batch/sequence 长度
  - 在 GPU 上计算可能更优（见下）

### 3. score·V 仍然使用 GPU 计算

- Softmax 后的 attention score 矩阵较小（batch × seq_len），数据量不大
- V 矩阵通常比 K 大（head_dim 维度），且 score·V 是 GEMM，Op/B 相对较高
- GPU 的高算力更适合这个阶段
- **分工**：PIM 做 Q·K^T → GPU 做 Softmax + score·V

```
┌─────────────────────────────────────────────────────────┐
│                    Attention 流水线                       │
│                                                         │
│  DRAM (2-bit K Cache)                                   │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────┐    score     ┌──────┐    output       │
│  │ PIM / Near-  │ ──────────→  │ GPU  │ ──────────→    │
│  │ Bank: Q·K^T  │   (FP16)     │Softmax│               │
│  │  (2-bit K)   │              │score·V│               │
│  └──────────────┘              └──────┘                │
│                                                         │
│  量化K,低带宽需求            高Op/B,适合GPU             │
└─────────────────────────────────────────────────────────┘
```

---

## Baseline 对比矩阵

| # | 方案 | KV 精度 | Q·K^T 处理器 | score·V 处理器 | 说明 |
|---|------|---------|-------------|---------------|------|
| B1 | **FP16 GPU** | 16-bit | GPU | GPU | 传统方案，全部 FP16 |
| B2 | **2-bit KV on GPU** | 2-bit | GPU | GPU | 仅量化 KV，仍在 GPU 算 |
| B3 | **2-bit KV on PIM** | 2-bit | PIM | PIM | 全 PIM，score·V 也在 PIM |
| B4 | **score·V on PIM** | 16-bit | PIM | PIM | 类似 Duplex 当前方案 |
| B5 | **score·V on GPU** | 16-bit | PIM | GPU | PIM 做 Q·K，GPU 做 score·V |
| **Idea** | **2-bit KV + PIM Q·K + GPU score·V** | 2-bit | PIM | GPU | **本文方案** |

---

## 需要模拟的 PIM 开销

### A. PIM 16-bit Float 开销

```
PIM 命令序列 (HBM3, ALL-RD):
  ALL-ACT → nRCDRD(19) → ALL-RD → nCL+nBL(21) → 数据到达
  延迟: 40 cycles (cold), 21 cycles (row hit)
  带宽: 8 banks × 256 bits = 2048 bits/cycle
  能耗: 从 DRAM bank 读 256bit → TSV → Logic die → GEMM unit
```

- 建模参数：`all_act_energy`, `all_read_energy`（已在 Ramulator HBM3 中配置）
- 带宽：PIM 4×（bank bundle 同时读）

### B. PIM 2-bit 开销

```
量化后的变化:
  1. 数据量减少 8×:
     - 每个 K cache token 从 128×128×2B = 32KB → 4KB
     - 相同 bank bundle 读操作可以覆盖 8× 更多的 token
  
  2. PIM 命令开销不变（ALL-ACT/ALL-RD 延迟相同）:
     - 但每个 ALL-RD 读出的有效数据量减少
     - 需要解量化逻辑（2-bit → FP16）在 Logic die 上
  
  3. 解量化开销:
     - 查表或简单乘法（scale + zero-point）
     - 在 Logic die GEMM unit 之前增加 dequant 阶段
     - 延迟增加约 1-2 cycles（流水线化）

  4. 能耗:
     - DRAM read 能耗不变（读同样的 256bit 物理数据）
     - 但有效数据密度增加 8× → 每有效 bit 能耗降低 8×
     - 解量化逻辑增加少量能耗
```

### 模拟方法

在 LLMSimulator 中需要修改/新增：

| 组件 | 修改内容 |
|------|---------|
| `MemoryObject` | 增加 `quantized_bit_width` 参数 |
| `PIM_KERNEL::GEMV` | 2-bit K 模式：每个 bundle 覆盖更多 token |
| `DRAMInterface` | 增加 dequant 能耗/延迟建模 |
| `mmap_controller` | 2-bit 地址映射调整 |
| `HBM3.cpp` | 可选：调整 ALL-RD 时序（如果 2-bit 模式减少传输量） |
| `config.yaml` | 增加 `kv_precision_bit` 参数 |

---

## 预期结论

1. **2-bit KV + PIM** 相比 FP16 GPU：
   - KV Cache 内存占用 ↓ 8× → 可支持更大 batch size 或更长序列
   - Q·K^T 带宽需求 ↓ 8× → PIM 优势放大

2. **score·V 仍在 GPU** 相比全部 PIM：
   - score·V 是 GEMM（高 Op/B），GPU 算力更匹配
   - 避免 PIM 处理高 Op/B 操作的效率损失

3. **与 Duplex baseline 对比**：
   - Duplex 只处理 FP16 KV cache 的 attention
   - 量化后 PIM 的带宽优势 × 量化压缩比 → 更大加速

---

## 实验设计

```
固定参数: model=mixtral, dev=4, input_len=512

变量:
  kv_bits:     [16, 2]
  qk_processor: [GPU, PIM]
  sv_processor: [GPU, PIM]

指标:
  - 吞吐 (tokens/s)
  - TTFT (time-to-first-token)
  - TBT (token-between-token)
  - DRAM 能耗
  - KV Cache 内存占用
```

---

## 风险与挑战

1. **2-bit 量化精度损失**：可能需要在 score 计算后做补偿（如 per-channel scale）
2. **解量化硬件开销**：Logic die 面积增加，需评估
3. **PIM 与 GPU 之间的 score 传输**：Q·K^T 结果需要从 PIM 搬回 GPU，增加通信开销
