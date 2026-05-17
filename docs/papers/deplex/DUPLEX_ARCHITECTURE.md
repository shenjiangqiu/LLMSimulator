# Duplex: LLM 推理的异构 PIM 架构

> 论文: *"Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching"*  
> arXiv: 2409.01141v1  
> 作者: Sungmin Yun, Kwanhee Kyung, Juhwan Cho, Jaewan Choi, Jongmin Kim, Byeongho Kim, Sukhan Lee, Kyomin Sohn, Jung Ho Ahn  
> 机构: Seoul National University & Samsung Electronics

---

## 一、核心问题

### LLM 推理的计算特征

LLM 推理中不同层的算数强度（Op/B）差异巨大：

| 层类型 | Op/B 范围 | GPU 利用率 | 瓶颈 |
|--------|----------|-----------|------|
| MoE（Expert FFN） | 1 ~ 32 | < 11% | 内存带宽 |
| Attention（GQA） | 4 ~ 8 | < 2.06% | 内存带宽 |
| FC（QKV 投影等） | > 32 | 较高 | 计算 |

传统 GPU 在处理低 Op/B 操作时利用率极低。而 Continuous Batching 下，**decode-only stage 占主导**（每个请求只有 1 个 prefill + L_out 个 decode），这些 stage 中 MoE 和 Attention 层占比最大。

### 为什么传统 PIM 不够用

```
Bank-PIM（在 DRAM die 内放处理单元）:
  ✓ 适合 Op/B < 1 的极端情况（利用 DRAM 内部高带宽）
  ✗ DRAM 工艺做逻辑单元面积开销巨大（占 die 面积 20%~27%）
  ✗ 算力不足以处理 Op/B 1~32 的操作
```

---

## 二、Duplex 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                      Duplex Device                        │
│                                                          │
│   ┌──────────────────┐       ┌──────────────────────┐    │
│   │      xPU         │       │   Logic-PIM (LowP)   │    │
│   │  (高 Op/B 处理器) │◄─────►│   (低 Op/B 处理器)    │    │
│   │                  │  HBM  │                      │    │
│   │  GPU / TPU       │  共享  │ ┌─────────────────┐  │    │
│   │  高算力           │  内存  │ │ GEMM Modules    │  │    │
│   │  标准 HBM 带宽    │       │ │ Softmax Module  │  │    │
│   └──────────────────┘       │ │ Activation Mod  │  │    │
│                               │ │ DRAM Controller │  │    │
│                               │ │ Buffer          │  │    │
│                               │ └─────────────────┘  │    │
│                               │    Logic Die (逻辑层) │    │
│                               └──────────┬───────────┘    │
│                                          │ ×4 TSVs        │
│                               ┌──────────┴───────────┐    │
│                               │   DRAM Dies (存储层)   │    │
│                               │ ┌─────────┬─────────┐│    │
│                               │ │Upper    │Lower    ││    │
│                               │ │Bank     │Bank     ││    │
│                               │ │Bundle   │Bundle   ││    │
│                               │ │(8 banks)│(8 banks)││    │
│                               │ └─────────┴─────────┘│    │
│                               └──────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 设计原则

| 组件 | 负责 | 特点 |
|------|------|------|
| **xPU** | FC 层、高 Op/B GEMM、prefill attention、all-reduce | 传统 GPU/TPU，高算力 |
| **Logic-PIM** | MoE expert FFN、decode attention（GEMV）、低 Op/B GEMM | 4× 内存带宽，Logic die 工艺 |

---

## 三、三大技术创新

### 3.1 Bank Bundle（银行捆绑）

**问题**：传统 HBM 中，一个 pseudo channel 的 16 个 bank 共享数据路径，一次只能读一个 bank。

**方案**：在 bank I/O 之间插入开关，分离数据路径，让多个 bank 同时读取。

```
传统 HBM（一次读 1 bank）:       Logic-PIM（一次读 8 banks）:
                                 
  Bank0 ─┐                        Upper Bundle (8 banks) ──── Logic-PIM
  Bank1 ─┤                         │  Bank0  Bank1  ...  Bank7
  Bank2 ─┼── 共享路径 ── 输出       │   ↓      ↓           ↓
  ...   ─┤                         │   ─────── 并行 ───────
  Bank15─┘                         │            ↓
                                   └────────── TSVs ──────── Logic Die

                                   Lower Bundle (8 banks) ──── xPU
                                   (xPU 可以同时独立访问)
```

**细节**：
- 16 banks 分为上下两组，各 8 banks，称为 **Bank Bundle**
- 同一 bank group 内连续读取延迟为 tCCD\_L（2× tCCD\_S），因此需要 8 个 bank 同时读才能实现 4× 带宽
- xPU 和 Logic-PIM 可以**同时访问不同的 bank bundle**，互不冲突

### 3.2 增加 TSV 数量

**问题**：Logic die 上放了处理单元，但没有更多数据通路，带宽和标准 HBM 一样。

**方案**：利用 TSV pitch 从 50μm → 22μm 的技术趋势，将 TSV 数量增加 4×。

```
传统 HBM:              Logic-PIM:
┌─────────────────┐    ┌─────────────────────┐
│  DRAM Dies      │    │  DRAM Dies          │
│  ██ ██ ██ ██    │    │  ████████████████   │
│  ██ ██ ██ ██    │    │  ████████████████   │
│  ██ ██ ██ ██    │    │  ████████████████   │
│  ██ ██ ██ ██    │    │  ████████████████   │
├─────────────────┤    ├─────────────────────┤
│  Logic Die      │    │  Logic Die          │
│  (仅 I/O 电路)   │    │  (I/O + 处理单元)    │
└─────────────────┘    └─────────────────────┘
  TSV pitch: 50μm         TSV pitch: 22μm
  带宽: 1×                 带宽: 4×
                          面积增加: ~9%
```

- 新增 TSV 放置在 **power TSV 区域**（而非数据 TSV 区域），以缩短 bank 到 TSV 的数据路径，降低能耗
- 避免改动 DRAM bank 布局（若增加每个 bank 的 prefetch size 会导致 77% 面积增长）

### 3.3 Logic Die 上的处理单元

**与传统 Bank-PIM/DRAM-die PIM 的对比**：

| 架构 | 处理单元位置 | 工艺 | 面积效率 | 适合 Op/B |
|------|------------|------|---------|----------|
| Bank-PIM | DRAM die 每个 bank | DRAM 工艺 | 低（占 die 20-27%） | < 1 |
| BankGroup-PIM | DRAM die | DRAM 工艺 | 低 | 1-8 |
| **Logic-PIM (Duplex)** | Logic die | **逻辑工艺** | **高** | **1-32** |

Logic die 上包含：
- **GEMM Modules**：执行矩阵乘法（MoE expert / Attention）
- **Softmax Module**：Attention 层的 softmax
- **Activation Module**：MoE 层的门控激活（SiLU 等）
- **DRAM Controller**：通过额外 TSV 从 bank bundle 取数
- **Operation Controller**：接收 xPU 发来的 PIM 指令（起始地址、GEMM 维度）
- **Buffer**：片上数据缓冲

---

## 四、Expert & Attention Co-processing

### 4.1 问题

各层之间有数据依赖，xPU 和 Logic-PIM 不能随意并行。Naive 方案（交替使用）利用率低。

### 4.2 Expert Co-processing

MoE 层中**不同 expert 之间没有数据依赖**，可以并行处理。

```
Expert Co-processing 流程:
1. Gate 计算完成后，统计每个 expert 被多少 token 选中
2. 查 lookup table（预先计算好的各 expert 在 xPU/Logic-PIM 上的处理时间）
3. 将 token 数少的 expert 分配给 Logic-PIM，多的分配给 xPU
4. xPU 发送 PIM 指令给 Logic-PIM 处理对应 expert
5. 所有 expert 完成后，xPU 执行一次 all-reduce 汇总结果
```

- **Decode-only stage**：Logic-PIM 处理大部分 expert
- **Mixed stage**（有新请求 prefill）：prefill 带来的大量 token 使 expert 的 Op/B 升高，xPU 处理高负载 expert

### 4.3 Attention Co-processing

Attention 操作在不同请求之间没有数据依赖。

- **Decode 请求的 attention** → Logic-PIM（每个请求独立 GEMV，低 Op/B）
- **Prefill 请求的 attention** → xPU（L_in 个 Q 共享 KV，高 Op/B GEMM）

### 4.4 内存分区

为防止 xPU 和 Logic-PIM 访问 bank bundle 冲突，将 HBM 内存分为 4 个区域：

| 内存分区 | 用途 | Bank Bundle |
|---------|------|-------------|
| 分区 0-2 | KV Cache（decode 用） | Bundle 0-2 |
| 分区 3 | Prefill 的 Q/K/V 矩阵 | Bundle 3 |
| 剩余 | 模型权重 + FC 层参数 | 任意 |

- MoE Expert FFN 权重逐一轮流分配到 4 个分区
- Co-processing 时，Logic-PIM 处理某个分区内的 expert，xPU 处理其他分区
- Prefill 的 K/V 矩阵在 attention 完成后迁移到 KV Cache 分区

---

## 五、与 LLMSimulator 代码的对应

| 论文概念 | 代码实现 |
|----------|----------|
| `xPU` (高 Op/B) | `ProcessorType::GPU` → `*ExecutionGPU()` |
| `Logic-PIM` (低 Op/B) | `ProcessorType::PIM` → `*ExecutionPIM()` |
| `Logic` (近存计算) | `ProcessorType::LOGIC` → `*ExecutionLogic()` |
| `bandwidth_x` (TSV 倍增) | `pim_x` / `logic_x` 配置（4/8/16） |
| `GEMV` (decode attention) | `PIM_KERNEL::GEMV` → 生成 `kMAC` 命令 |
| `Expert Co-processing` | `parallel_execution` 模式 |
| `Lookup table` 决策 | `LayerInfo.processor_type` + `executePType()` 遍历 |
| `Bank Bundle` 地址映射 | `MemoryObject::getAddrVec()` |
| `All-reduce` 汇总 | `TopModuleGraph::set_pop_status()` |
| `kMAC` → AllRead | `PIMController::get_type()` |

### 核心调用路径

```
config.yaml
  processor_type: [GPU, PIM]
  pim_x: 16
  logic_x: 4
      │
      ▼
Module::forward() → LayerInfo.processor_type
      │
      ▼
Executor::executePType(ProcessorType::PIM)
  → device->dram_interface->setPIMHWConfig(PIM, bandwidth_x=16)
  → *ExecutionPIM()
      → issueRamulator(device, ..., DRAMRequestType::kGEMV/Read/Write, tensor)
          → DRAMInterface::HandleRequest()
              → PIM_KERNEL::GEMV(..., pim_hw_config)
                  → 根据 type==PIM 和 bandwidth_x 生成 kMAC
              → PIMController::send()
                  → kMAC → AllRead → MemorySystem
```

---

## 六、性能数据

与纯 GPU 系统（4×GPU）对比，Mixtral 模型：

| 指标 | 提升 |
|------|------|
| 吞吐量 | **最高 2.67×** |
| 能耗 | **降低 42.0%** |
| 延迟（TBT/ETE） | 显著降低 |

### EDAP 对比（Energy-Delay-Area Product）

| Op/B | 最优架构 |
|------|---------|
| < 1 | Bank-PIM（DRAM die，最高内部带宽） |
| 1 ~ 8 | BankGroup-PIM 或 Logic-PIM |
| > 8 | **Logic-PIM (Duplex)**（最佳 EDAP） |

---

## 七、关键术语

| 术语 | 全称 | 含义 |
|------|------|------|
| MoE | Mixture of Experts | 混合专家，每个 token 只过 top-k 个 expert FFN |
| GQA | Grouped Query Attention | 多个 head 共享 K/V，减少内存带宽需求 |
| Continuous Batching | 连续批处理 | 以 stage 级别调度，而非 request 级别 |
| Op/B | Operations per Byte | 算数强度，每字节内存访问对应的计算量 |
| Decode-only stage | 纯解码阶段 | batch 中所有请求都在 decode |
| Mixed stage | 混合阶段 | 有新请求 prefill 加入 |
| Bank Bundle | 银行捆绑 | 8 个 bank 为一组同时操作 |
| TSV | Through-Silicon Via | 穿过硅通孔，连接 DRAM die 和 Logic die |
| tCCD\_L / tCCD\_S | CAS-to-CAS Delay | 同 bank group / 跨 bank group 的命令间隔 |
