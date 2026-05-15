# DRAMRequestType / PIMCommandType / DRAMCommandType 关系

三层枚举类型构成从高层语义到底层信号的翻译链：

```
LLM Layer                 PIM_KERNEL             Ramulator
(高层语义)                (PIM ISA)              (DRAM 信号)
    │                         │                      │
    ▼                         ▼                      ▼
DRAMRequestType    ──→   PIMCommandType    ──→   DRAMCommandType
"我要做什么操作"          "PIM 执行什么指令"       "DRAM 发出什么信号"
```

---

## 第 1 层：DRAMRequestType — 高层操作语义

**文件**：`src/dram/dram_type.h:12`

```cpp
enum class DRAMRequestType {
  kRead = 0,          // 普通读
  kWrite = 1,         // 普通写
  kMove = 2,          // 数据搬运（未启用）
  kMult = 3,          // 逐元素乘（未启用）
  kAdd = 4,           // 逐元素加（未启用）
  kMAD = 5,           // 乘加（未启用）
  kPMult = 6,         // PIM 乘（未启用）
  kCMult = 7,         // 跨 PIM 乘（未启用）
  kCAdd = 8,          // 跨 PIM 加（未启用）
  kCMAD = 9,          // 跨 PIM 乘加（未启用）
  kTensor = 10,       // Tensor 操作（未启用）
  kTensor_Square = 11,// Tensor 平方（未启用）
  kModup_Evkmult = 12,// 模上乘（未启用）
  kModDownEpilogue=13,// 模下尾声（未启用）
  kPMult_Accum = 14,  // PIM 乘累加（未启用）
  kCMult_Accum = 15,  // 跨 PIM 乘累加（未启用）
  kGEMV = 16,         // 矩阵向量乘 ← 唯一实际启用的计算类型
  kMAX,
};
```

- **谁生成**：`issueRamulator()` 构造 `DRAMRequest` 时指定
- **谁消费**：`PIM_KERNEL::init()` 注册的 kernel 函数表
- **当前状态**：仅 `kRead`(0)、`kWrite`(1)、`kGEMV`(16) 有对应 kernel

### 注册表

```cpp
// pim_kernel.cpp
kernel[(int)DRAMRequestType::kRead]  = &Read_kernel;
kernel[(int)DRAMRequestType::kWrite] = &Write_kernel;
kernel[(int)DRAMRequestType::kGEMV]  = &GEMV;
// 其余 14 种: 注释掉，未实现
```

---

## 第 2 层：PIMCommandType — PIM 单元 ISA

**文件**：`src/dram/dram_type.h:33`

```cpp
enum class PIMCommandType {
  kAdd = 0,           // r0 = r1 + r2
  kSub,               // r0 = r1 - r2
  kMult,              // r0 = r1 * r2
  kMAC,               // r0 = r1 * r2 + r3   ← 核心指令
  kDRAM2RF,           // 加载: DRAM → Register File
  kRF2DRAM,           // 存储: Register File → DRAM
  kRead,              // 普通读
  kWrite,             // 普通写
  kMAX,
};
```

- **谁生成**：`PIM_KERNEL`（`GEMV()`, `Read_kernel()`, `Write_kernel()`）
- **谁消费**：`PIMController::get_type()` 映射为 Ramulator `Request::Type`
- **当前实际使用**：`kRead`、`kWrite`、`kMAC`

### 到 Ramulator Request 的映射

```cpp
// PIMController::get_type()
kAdd/kSub/kMult/kMAC/kDRAM2RF  →  Request::Type::AllRead   (2)
kRF2DRAM                        →  Request::Type::AllWrite  (3)
kRead                           →  Request::Type::Read      (0)
kWrite                          →  Request::Type::Write     (1)
```

---

## 第 3 层：DRAMCommandType — DRAM 物理信号

**文件**：`src/dram/dram_type.h:45`

```cpp
enum class DRAMCommandType {
  kACT = 0,           // 激活行（单个 bank）
  kPRE,               // 预充电（单个 bank）
  kPREA,              // 预充电（所有 bank）
  kALL_PRE,           // PIM: 全部 bank 预充电
  kREAD,              // 读（单个 bank）
  kWRITE,             // 写（单个 bank）
  kALL_ACT,           // PIM: 全部 bank 激活
  kALL_READ,          // PIM: 全部 bank 读
  kALL_WRITE,         // PIM: 全部 bank 写
  kREF,               // 刷新
  kMAX,
};
```

- **谁生成**：`IDRAM::issue_command()` 被 DRAM Controller 调用
- **语义**：DRAM Controller 向 DRAM 芯片发出的实际电气信号

### Ramulator 侧的对应

```cpp
// HBM3.cpp m_request_translations
{"read", "RD"}       →  Request::Type::Read → 最终 DRAM 命令: ACT + RD + PRE
{"write", "WR"}      →  Request::Type::Write
{"all-read", "ALL-RD"} →  Request::Type::AllRead → 最终 DRAM 命令: ALL-ACT + ALL-RD + ALL-PRE
{"all-write", "ALL-WR"}→  Request::Type::AllWrite
```

---

## 完整映射链（以 kGEMV → kMAC 为例）

```
1. issueRamulator(device, ATTENTION_GEN, PIM, kGEMV, kSrc, tensor)
   DRAMRequest { type = DRAMRequestType::kGEMV (16) }

2. PIM_KERNEL::GEMV(pimrequest, kGEMV, operands, {PIM, bw=16})
   遍历 16 个 bundle:
     PIMCommand { pimcmd_type = PIMCommandType::kMAC, addr_vec = {...} }
     PIMCommand { pimcmd_type = PIMCommandType::kMAC, addr_vec = {...} }
     ... (16 个 kMAC)

3. PIMController::send()
   for each PIMCommand:
     get_type(kMAC) → Request::Type::AllRead (2)
     memory_system->send({addr_vec, AllRead, cmd, &set_end_time})

4. PIM_DRAM_system::send()
   m_addr_mapper->apply(req)
   路由到 m_controllers[channel_id]->send(req)

5. PIMDRAMController::send()
   req.final_command = m_request_translations[AllRead]
     → "all-read" → "ALL-RD" (index 9)

6. PIMDRAMController::tick() → schedule_request() → issue_command()
   get_preq_command(ALL-RD, addr_vec)
     bank closed?  → DRAMCommandType::kALL_ACT
     bank opened?  → 直接 ALL-RD
   
   issue_command(kALL_ACT, addr)   // 激活所有 bank 的行
   issue_command(kALL_RD, addr)    // 读取

7. updateStatus() — 收集统计
   issued_dram_cmd[kALL_ACT]++
   issued_dram_cmd[kALL_READ]++
```

---

## 关键文件

| 层 | 定义 | 翻译/映射 |
|----|------|----------|
| DRAMRequestType | `src/dram/dram_type.h:12` | `PIM_KERNEL::init()` (pim_kernel.cpp) |
| PIMCommandType | `src/dram/dram_type.h:33` | `PIMController::get_type()` (PIM_controller.cpp) |
| DRAMCommandType | `src/dram/dram_type.h:45` | `HBM3::m_request_translations` (HBM3.cpp) |
| Ramulator Request::Type | `ramulator2/src/base/request.h` | `m_request_translations` → 最终 DRAM 命令 |
