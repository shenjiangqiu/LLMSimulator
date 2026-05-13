# LLMSimulator PIM 架构文档

## 概述

LLMSimulator 支持三种处理器类型混合执行神经网络推理：

| 处理器类型 | 枚举值 | 说明 |
|-----------|--------|------|
| `GPU` | `ProcessorType::GPU` | 传统 GPU，Read/Write 数据 |
| `LOGIC` | `ProcessorType::LOGIC` | 近存计算逻辑单元，通过 GEMV 操作 |
| `PIM` | `ProcessorType::PIM` | 存内计算（Processing-In-Memory），直接在 DRAM 内执行 MAC 操作 |

系统通过 Ramulator2 进行周期精确的 DRAM 时序模拟，PIM 核心逻辑位于 `src/dram/` 目录下的 `pimkernel` 和 `ramulator2` 子模块中。

---

## 整体调用流程

```
config.yaml                     Module::forward()              Executor::executePType()
┌──────────────┐    ┌──────────────────────────────┐    ┌──────────────────────────┐
│processor_type│───>│ 根据 parallel_execution /    │───>│ 函数表按 ProcessorType   │
│high/low type │    │ hetero_subbatch / optimal    │    │ 索引到 *ExecutionPIM()   │
│logic_x/pim_x │    │ 决定 LayerInfo.processor_type│    │ 设置 bandwidth_x         │
└──────────────┘    └──────────────────────────────┘    └──────────────────────────┘
                                                                │
                                                                ▼
                         issueRamulator()  ← 构造 DRAMRequest
                                │
                                ▼
                    Device::run_ramulator(dram_request)
                                │
                                ▼
                  DRAMInterface::HandleRequest()
                     │
         ┌───────────┼───────────────┐
         ▼           ▼               ▼
  GeneratePIMCommand  SendRequest    run() + updateStatus()
  (PIM_KERNEL 翻译)   (发给 Frontend)   (驱动时序 + 收集统计)
         │               │
         ▼               ▼
    PIM_KERNEL:      PIMController::send()
  Read_kernel        ┌─────────────────────┐
  Write_kernel       │ PIMCommand → Request│
  GEMV               │ kMAC → AllRead      │
                     │ kRead → Read        │
                     │ kWrite → Write      │
                     └─────────────────────┘
                           │
                           ▼
                    MemorySystem::send()
                           │
                           ▼
                    DRAMController (周期精确模拟)
```

---

## PIM 命令翻译 (PIM_KERNEL)

### 内核注册

文件：`src/dram/pimkernel/pim_kernel.cpp`

```cpp
void init(std::vector<std::function<pim_kernel_ptr>>& kernel) {
  kernel.resize((int)DRAMRequestType::kMAX);
  kernel[(int)DRAMRequestType::kRead]  = &Read_kernel;
  kernel[(int)DRAMRequestType::kWrite] = &Write_kernel;
  kernel[(int)DRAMRequestType::kGEMV]  = &GEMV;
  // 以下内核暂未启用：
  // kMove, kMult, kAdd, kMAD, kPMult, kCMult, kCAdd, kCMAD,
  // kTensor, kTensor_Square, kModup_Evkmult, kModDownEpilogue,
  // kPMult_Accum, kCMult_Accum
}
```

当前启用的内核：

| DRAMRequestType | Kernel 函数 | 文件 | 生成 PIM 命令 |
|----------------|-------------|------|--------------|
| `kRead` | `Read_kernel` | `Read.cpp` | `kRead` |
| `kWrite` | `Write_kernel` | `Write.cpp` | `kWrite` |
| `kGEMV` | `GEMV` | `GEMV.cpp` | `kMAC`（取决于处理器类型和带宽） |

### Read_kernel 和 Write_kernel

```cpp
// Read.cpp — 遍历 MemoryObject 的每个 bundle，生成 kRead 命令
void Read_kernel(PIMRequest& pim_request, ...) {
  auto read_operand = get_operand(operand, PIMOperandType::kDRAM);
  for (auto opnd : read_operand) {
    for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
      addr_vec = opnd->getAddrVec(bundle_idx, pim_hw_config.type);
      if (addr_vec.at(0) == 0) {
        pim_request.AddCommand(PIMCommand(
          PIMCommandType::kRead, PIMOperandType::kDRAM, addr_vec,
          &pim_request, dramreq_type
        ));
      }
    }
  }
}
```

Write_kernel 逻辑相同，生成 `PIMCommandType::kWrite`。

### GEMV_kernel

GEMV 是最核心的 PIM 操作。它根据 `ProcessorType` 和 `bandwidth_x` 决定如何生成 `kMAC` 命令：

```
GEMV(PIMRequest, DRAMRequestType, Operand, PIMHWConfig)
  │
  ├─ ProcessorType::PIM
  │   ├─ bandwidth_x == 16 → 所有 bundle 生成 kMAC（无限制）
  │   ├─ bandwidth_x == 8  → addr[4] % 2 == 0 的 bundle 生成 kMAC
  │   └─ bandwidth_x == 4  → 所有 bundle 生成 kMAC（无 bank 限制）
  │
  └─ ProcessorType::LOGIC
      └─ bandwidth_x == 4  → addr[4] % 2 == 0 的 bundle 生成 kMAC
```

`bandwidth_x` 决定了 PIM 单元的数据通路宽度，影响地址对齐和 bank 选择逻辑。

### PIM 命令到 DRAM Request 的二次映射

文件：`src/dram/ramulator2/src/frontend/impl/PIM_controller.cpp`

```cpp
int get_type(PIMCommandType type) {
  switch (type) {
    case kAdd: case kSub: case kMult: case kMAC: case kDRAM2RF:
      return (int)Request::Type::AllRead;    // PIM 计算操作 → DRAM AllRead
    case kRF2DRAM:
      return (int)Request::Type::AllWrite;   // RF 写回 DRAM → AllWrite
    case kRead:
      return (int)Request::Type::Read;       // 普通读
    case kWrite:
      return (int)Request::Type::Write;      // 普通写
  }
}
```

---

## 处理器类型决策机制

### 配置入口

文件：`config.yaml`

```yaml
system:
  processor_type: GPU          # 默认处理器类型（可多选，如 [GPU, PIM]）
  optimization:
    parallel_execution: off    # 并行执行：Gen→低端, Sum→高端
    hetero_subbatch: off       # 异构子批次：全部用高端
```

`SystemConfig` 类（`src/hardware/hardware_config.h`）：

```cpp
class SystemConfig {
  ProcessorType high_processor_type = ProcessorType::GPU;    // "高"处理器
  ProcessorType low_processor_type  = ProcessorType::LOGIC;  // "低"处理器
  std::vector<ProcessorType> processor_type;                 // 默认类型列表
  int logic_x = 4;  // LOGIC 单元带宽倍数
  int pim_x   = 16; // PIM 单元带宽倍数
};
```

### 裁决逻辑

每个 Module（Linear、Activation、Attention 等）在 `forward()` 中设置 `LayerInfo.processor_type`：

```cpp
LayerInfo info;
info.processor_type = device->config.processor_type;  // 默认

if (input->parallel_execution) {
    if (input->isPerformHigh())
        info.processor_type = {device->config.high_processor_type};  // Sum→GPU
    else
        info.processor_type = {device->config.low_processor_type};   // Gen→LOGIC/PIM
}
else if (device->config.hetero_subbatch)
    info.processor_type = {device->config.high_processor_type};      // 全用高端
else if (device->config.processor_type.size() != 1)
    // 多类型时遍历取性能最优
```

### 并行执行模式

当 `parallel_execution = on` 时，系统将模型分为高低两条流水线：

| 阶段 | 处理器 | 原因 |
|------|--------|------|
| **Gen**（生成阶段） | `low_processor_type`（如 LOGIC/PIM） | 内存密集，适合近存/PIM |
| **Sum**（汇总阶段） | `high_processor_type`（如 GPU） | 计算密集，适合 GPU |

### Executor 函数表分发

`Executor::init()` 为每种处理器类型注册对应的执行函数：

```cpp
void Executor::initLinear() {
  linear_function_ramulator_ptr[(int)ProcessorType::GPU]   = LinearExecutionGPU;
  linear_function_ramulator_ptr[(int)ProcessorType::LOGIC] = LinearExecutionLogic;
  linear_function_ramulator_ptr[(int)ProcessorType::PIM]   = LinearExecutionPIM;
}
// initActivation / initAttentionGen / initAttentionSum 等同理
```

---

## 关键数据结构

### DRAMRequest

高层 DRAM 请求，包含操作类型和操作数（MemoryObject）。

```cpp
class DRAMRequest {
  DRAMRequestType type;                                      // kRead/kWrite/kGEMV
  std::unordered_map<PIMOperandType, std::vector<MemoryObject::Ptr>> operands_;
};
```

### PIMRequest / PIMCommand

经过 PIM_KERNEL 翻译后的底层 PIM 命令队列。

```cpp
class PIMRequest {
  std::list<PIMCommand> command_queue;  // PIM 命令序列
  std::vector<counter_t> issued_pim_cmd;   // 各 PIM 命令计数
  std::vector<counter_t> issued_dram_cmd;  // 各 DRAM 命令计数（ACT/RD/WR/REF）
  cycle_t start, end;                      // 起止周期
};

class PIMCommand {
  DRAMRequestType dramreq_type;    // 原始高层请求类型
  PIMCommandType  pimcmd_type;     // kRead/kWrite/kMAC/kDRAM2RF/kRF2DRAM
  PIMOperandType  op_type;         // kDRAM/kRF/kSrc/kEvk/kModUp
  AddrVec_t       addr_vec;        // 地址向量 (channel, rank, bank, row, col, ...)
};
```

### PIMHWConfig

传递给 PIM_KERNEL 的硬件配置上下文。

```cpp
struct PIMHWConfig {
  ProcessorType type;     // GPU / LOGIC / PIM
  int bandwidth_x;        // 带宽倍数（4/8/16），影响 MAC 命令生成策略
};
```

### Operand 类型

| PIMOperandType | 含义 |
|---------------|------|
| `kDRAM` | DRAM 中的操作数（Read/Write kernel 使用） |
| `kSrc` | 源操作数（GEMV kernel 使用） |
| `kRF` | Register File 中的操作数 |
| `kEvk` | 评估密钥操作数 |
| `kModUp` | 模上操作数 |

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `src/dram/dram_interface.h/.cpp` | PIM 调用主入口，HandleRequest / GeneratePIMCommand |
| `src/dram/pimkernel/pim_kernel.h/.cpp` | PIM 内核注册表，init() 函数 |
| `src/dram/pimkernel/Read.cpp` | Read_kernel：kRead → PIM 读命令 |
| `src/dram/pimkernel/Write.cpp` | Write_kernel：kWrite → PIM 写命令 |
| `src/dram/pimkernel/GEMV.cpp` | GEMV：kGEMV → kMAC 命令（PIM/LOGIC 分支） |
| `src/dram/pim_request.h` | PIMRequest / PIMCommand 数据结构 |
| `src/dram/dram_request.h` | DRAMRequest 数据结构 |
| `src/dram/dram_type.h` | DRAMRequestType / PIMCommandType / PIMOperandType 枚举 |
| `src/dram/ramulator2/src/frontend/impl/PIM_controller.cpp` | PIMController：PIM 命令 → DRAM Request 映射 |
| `src/dram/ramulator2/src/frontend/frontend.h` | IFrontEnd 接口定义 |
| `src/hardware/executor.h/.cpp` | Executor：处理器类型分发 + 函数表 |
| `src/hardware/device.h/.cpp` | Device：run_ramulator / dram_interface |
| `src/hardware/layer_impl.h/.cpp` | issueRamulator / getIdealMemoryStatus |
| `src/hardware/hardware_config.h` | SystemConfig：processor_type / pim_x / logic_x 配置 |
| `src/hardware/base.h` | LayerInfo 结构（含 processor_type） |
| `src/module/linear.cpp` | Linear::forward()：处理器类型裁决 |
| `src/module/activation.cpp` | Activation::forward()：处理器类型裁决 |
| `src/module/attention.cpp` | SelfAttention::forward()：处理器类型裁决 |
| `src/dram/memory_object.h/.cpp` | MemoryObject：getAddrVec / getBundleSize |
| `config.yaml` | 用户配置文件 |
