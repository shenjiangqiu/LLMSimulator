# PIM 命令时序详解

一个 `kMAC` PIMCommand 从进入 MemorySystem 到完成的完整时序。

---

## 命令映射链

```
kMAC (PIMCommand)
  →  Request::Type::AllRead (= 2)
    →  "all-read" → final_command = ALL-RD (= 9)
```

HBM3 中 PIM 使用特殊的 **ALL-** 系列命令，对所有 bank 同时操作：

| 命令 | 全称 | 作用 | 作用域 |
|------|------|------|--------|
| `ALL-ACT` | All-Bank Activate | 激活所有 bank 的同一行 | rank |
| `ALL-RD` | All-Bank Read | 从所有 bank 读数据 | rank |
| `ALL-WR` | All-Bank Write | 向所有 bank 写数据 | rank |
| `ALL-PRE` | All-Bank Precharge | 关闭所有 bank 的当前行 | rank |

---

## 命令序列状态机

一个 `ALL-RD` 的前置条件由 `RequirePIMRowOpen` 决定：

```
当前 Bank 状态        前置命令            下一状态
─────────────────────────────────────────────────
Closed            →  ALL-ACT           → Opened (激活行)
Opened (行命中)    →  ALL-RD (直接)      → Opened (保持)
Opened (行冲突)    →  ALL-PRE           → Closed
                     → ALL-ACT          → Opened
                     → ALL-RD           → Opened
```

---

## HBM3 关键时序参数

以 **HBM3_5.2Gbps** (tCK = 1300 ps = 0.769 GHz) 为例：

| 参数 | 值 (cycles) | 含义 |
|------|------------|------|
| `nRCDRD` | 19 | ACT → RD 最小间隔 |
| `nRCDWR` | 19 | ACT → WR 最小间隔 |
| `nCL` | 19 | CAS 延迟（RD 命令到数据就绪） |
| `nBL` | 2 | Burst 长度 |
| `nRP` | 19 | PRE → ACT 最小间隔 |
| `nRAS` | 45 | ACT → PRE 最小间隔（行激活最短时间） |
| `nRTPS` | 6 | RD → PRE 最小间隔 |
| `nCCDL` | 4 | 跨 bank group 的 CAS-to-CAS 延迟 |
| `nRC` | 63 | 同一 bank 的 ACT-to-ACT 最小间隔 |
| **Read Latency** | **21** | `nCL + nBL`（RD 到数据到达控制器） |

---

## 三种场景的完整时序

### 场景 1：Bank 初始关闭（Cold Start）

```
Cycle │ DRAM 命令         │ 时序约束
──────┼───────────────────┼──────────────────────────
  0   │ ALL-ACT 发出      │
  1~18│ 等待 nRCDRD       │ nRCDRD = 19 cycles
 19   │ ALL-RD 发出       │ (ACT+19 = 19, ≥ nRCDRD ✓)
 20~39│ 等待 Read Latency │ nCL + nBL = 21 cycles
 40   │ 数据到达控制器    │ req.depart = 40
      │ callback 触发      │ set_end_time(req)
```

**总延迟：40 cycles (52 ns @ 5.2Gbps)**

时间分解：
- `nRCDRD`：19 cycles — 行激活时间（信号从 bank 传到 I/O）
- `nCL + nBL`：21 cycles — 读延迟（列选通 + 数据突发传输）

---

### 场景 2：Bank 已打开且行命中（Row Hit）

```
Cycle │ DRAM 命令         │ 时序约束
──────┼───────────────────┼──────────────────────────
  0   │ ALL-RD 发出       │ (行已激活，直接读)
  1~20│ 等待 Read Latency │ nCL + nBL = 21 cycles
 21   │ 数据到达控制器    │ req.depart = 21
      │ callback 触发      │ set_end_time(req)
```

**总延迟：21 cycles (27.3 ns)**

这是最快路径：不需要 ACTIVATE，直接读已打开的行。

---

### 场景 3：Bank 已打开但行冲突（Row Conflict）

```
Cycle │ DRAM 命令         │ 时序约束
──────┼───────────────────┼──────────────────────────
  0   │ ALL-PRE 发出      │ (关闭当前行)
  1~18│ 等待 nRP          │ nRP = 19 cycles
 19   │ ALL-ACT 发出      │ (PRE+19 = 19, ≥ nRP ✓)
 20~37│ 等待 nRCDRD       │ nRCDRD = 19 cycles
 38   │ ALL-RD 发出       │ (ACT+19 = 38, ≥ nRCDRD ✓)
 39~58│ 等待 Read Latency │ nCL + nBL = 21 cycles
 59   │ 数据到达控制器    │ req.depart = 59
```

**总延迟：59 cycles (76.7 ns)**

最慢路径，需要先关闭旧行再激活新行。

---

## 连续多个 kMAC 的流水线

当多个 kMAC 命中同一行时，可以利用 `nCCDL` 进行流水线：

```
Cycle │ 命令序列 (3 个 kMAC, 全部 Row Hit)
──────┼─────────────────────────────────────────
  0   │ ALL-RD #1 发出
  4   │ ALL-RD #2 发出   (nCCDL = 4 ✓)
  8   │ ALL-RD #3 发出   (nCCDL = 4 ✓)
  21  │ #1 数据到达
  25  │ #2 数据到达
  29  │ #3 数据到达
```

- **吞吐率**：每 4 cycles 处理一个 kMAC
- **延迟**：首个 kMAC 仍为 21 cycles
- 16 个 kMAC 连续命中同行的总时间：`21 + 15×4 = 81 cycles`

---

## 与普通 Read 的对比

普通 `kRead → Type::Read → RD` 使用单个 bank 的命令：

| 属性 | PIM (ALL-RD) | 普通 (RD) |
|------|-------------|----------|
| 作用域 | 所有 bank | 单个 bank |
| 数据带宽 | 8 banks × 256 bits | 1 bank × 256 bits |
| Row Hit 延迟 | 21 cycles | 21 cycles |
| 连续吞吐 | nCCDL = 4 | nCCDS = 2 |
| 数据量 | 2048 bits/cycle | 256 bits/cycle |
| 带宽倍数 | 8× (同一行) | 1× |

PIM 用 ALL-RD 一次性从 8 个 bank（bank bundle）同时读取，获得 8× 带宽。但 ALL-RD 间的间隔是 `nCCDL = 4`，而普通 RD 间的间隔是 `nCCDS = 2`，这是因为 bank bundle 内的 bank 来自同一 bank group，需要遵守更长的时序约束。

---

## 关键文件

| 文件 | 内容 |
|------|------|
| `ramulator2/src/dram/impl/HBM3.cpp:29-35` | HBM3 时序参数预设 |
| `ramulator2/src/dram/impl/HBM3.cpp:99-110` | `m_request_translations`（AllRead → ALL-RD） |
| `ramulator2/src/dram/impl/HBM3.cpp:331-352` | PIM 命令时序约束（ALL-ACT/ALL-RD/ALL-PRE 间的关系） |
| `ramulator2/src/dram/impl/HBM3.cpp:454-464` | `set_preqs()`（ALL-RD 前置条件 = RequirePIMRowOpen） |
| `ramulator2/src/dram/lambdas/preq.h:29-44` | `RequirePIMRowOpen`（Closed→ALL-ACT, Opened→命中则直接 RD, 冲突则 ALL-PRE） |
| `ramulator2/src/dram_controller/impl/PIM_dram_controller.cpp` | PIMDRAMController 调度逻辑 |
