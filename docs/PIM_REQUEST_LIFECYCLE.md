# PIM 请求完整生命周期

以一个具体案例追踪：**Decode 阶段 Attention 的 GEMV 操作**，处理器类型 **PIM**，`bandwidth_x = 16`。

---

## 全程概览

```
┌─ 第 1 步 ──────────────────────────────────────────────────────────┐
│ AttentionGenExecutionPIM()                                          │
│   → issueRamulator(device, ATTENTION_GEN, PIM, kGEMV, kSrc, tensor)│
│     → DRAMRequest::Create(kGEMV)                                    │
│     → dram_request->AddOperand(tensor->getMemoryObject(), kSrc)     │
│     → device->run_ramulator(dram_request)                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─ 第 2 步 ──────────────────────────────────────────────────────────┐
│ Device::run_ramulator()                                             │
│   → dram_interface->HandleRequest({dram_request}, 0)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─ 第 3 步 ──────────────────────────────────────────────────────────┐
│ DRAMInterface::HandleRequest()                                      │
│   → GeneratePIMCommand(dram_req, pimrequest)   // 翻译             │
│   → SendRequest(pimrequest)                     // 发送到 Frontend   │
│   → run()                                       // 驱动时序         │
│   → updateStatus(pimrequest)                    // 收集统计         │
└─────────────────────────────────────────────────────────────────────┘
          │                    │                    │
┌─ 第 3a 步 ─────────┐  ┌─ 第 3b 步 ────────┐  ┌─ 第 3c 步 ─────────┐
│ PIM_KERNEL::GEMV()  │  │ PIMController::send│  │ MemorySystem::tick()│
│ 生成 PIMCommand 队列 │  │ PIM 命令→DRAM Req │  │ 逐周期驱动控制器     │
└─────────────────────┘  └───────────────────┘  └────────────────────┘
```

---

## 第 1 步：请求生成

### 1.1 入口

文件：`src/hardware/attention_gen_impl.cpp`

```cpp
ExecStatus AttentionGenExecutionPIM(Device_Ptr device, ...) {
    // k_cache 是 Key Cache tensor
    exec_status += issueRamulator(
        device,                       // Device 实例
        LayerType::ATTENTION_GEN,     // 层类型
        ProcessorType::PIM,           // 用 PIM 处理器
        DRAMRequestType::kGEMV,       // GEMV 操作（decode attention）
        PIMOperandType::kSrc,         // 操作数类型：源数据
        k_cache                       // 目标 Tensor
    );
}
```

### 1.2 issueRamulator

文件：`src/hardware/layer_impl.cpp`

```cpp
ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type,
                          Tensor_Ptr tensor) {
    // 1. 构造缓存 Key：{层类型, 处理器类型, 请求类型, 数据大小}
    CacheKey key = std::make_tuple(layer_type, processor_type,
                                   dram_request_type, tensor->getSize());
    
    // 2. 查缓存，命中则直接返回
    if (!device->checkExecutionCache(exec_status, key)) {
        // 3. 创建 DRAMRequest
        DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
        //    dram_request->type = DRAMRequestType::kGEMV
        
        // 4. 绑定操作数：将 Tensor 的 MemoryObject 绑定到 kSrc
        dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
        //    operands_[kSrc] = { MemoryObject }
        
        // 5. 送到 Device 执行
        device->run_ramulator(dram_request);
        
        // 6. 获取执行状态（时间、能耗等）
        exec_status = device->dram_interface->getExecStatus();
        
        // 7. 写入缓存
        device->addExecutionCache(exec_status, key);
    }
    return exec_status;
}
```

### 1.3 Device::run_ramulator

文件：`src/hardware/device.cpp`

```cpp
void Device::run_ramulator(DRAMRequest_Ptr dram_request) {
    std::list<DRAMRequest::Ptr> request;
    request.push_back(dram_request);
    dram_interface->HandleRequest(request, 0);
}
```

---

## 第 2 步：PIM_KERNEL 翻译（DRAMRequest → PIMRequest）

### 2.1 HandleRequest

文件：`src/dram/dram_interface.cpp`

```cpp
void DRAMInterface::HandleRequest(
    const std::list<DRAMRequest::Ptr>& requests,
    cycle_t start_time_cycle) {
    
    resetCounter();
    
    for (auto& dram_req : requests) {
        PIMRequest pimrequest;                    // 空 PIMRequest
        
        // === 2.2 翻译 ===
        GeneratePIMCommand(dram_req, pimrequest);
        
        // === 2.3 发送 ===
        SendRequest(pimrequest);
        
        // === 2.4 驱动时序模拟 ===
        run();   // 循环 tick() 直到所有请求完成
        
        // === 2.5 收集统计 ===
        updateStatus(pimrequest);
    }
}
```

### 2.2 GeneratePIMCommand

```cpp
PIMRequest& DRAMInterface::GeneratePIMCommand(
    const DRAMRequest::Ptr request,
    PIMRequest& pimrequest) const {
    
    DRAMRequestType type = request->GetType();
    // type = DRAMRequestType::kGEMV (= 16)
    
    // 查表调用对应 kernel:
    // kernel[16] = &GEMV  (在 PIM_KERNEL::init() 中注册)
    kernel[int(type)](pimrequest, type, request->operands_, pim_hw_config);
    //                     │          │          │              │
    //                     │          │          │              └─ {type=PIM, bandwidth_x=16}
    //                     │          │          └─ operands_[kSrc] = {MemoryObject}
    //                     │          └─ kGEMV
    //                     └─ 输出：pimrequest.command_queue 被填充
}
```

### 2.3 GEMV Kernel 翻译过程

文件：`src/dram/pimkernel/GEMV.cpp`

```
输入: PIMHWConfig { type = PIM, bandwidth_x = 16 }

1. 从 operands_ 中取出 kSrc 操作数
   auto src = get_operand(operand, PIMOperandType::kSrc);
   MemoryObject::Ptr opnd = src.at(0);
   
2. MemoryObject 结构:
   opnd->address = 某个逻辑地址
   opnd->size = tensor 字节大小
   opnd->num_bundle = size / granul  (如 4096 / 256 = 16 个 bundle)

3. 遍历每个 bundle:
   for (int bundle_idx = 0; bundle_idx < 16; bundle_idx++) {
       
       // 获取物理地址向量
       addr_vec = opnd->getAddrVec(bundle_idx);
       // addr_vec = {channel, rank, bankgroup, bank, row, col, ...}
       
       // PIM + bandwidth_x=16: 所有 bundle 都生成 kMAC
       if (addr_vec[0] == 0 && addr_vec[1] == 0 &&
           addr_vec[2] == 0 && addr_vec[3] == 0 && addr_vec[4] == 0) {
           
           // 创建 PIMCommand 并加入队列
           pim_request.AddCommand(PIMCommand(
               PIMCommandType::kMAC,      // PIM 乘累加
               PIMOperandType::kSrc,      // 源操作数
               addr_vec,                  // 物理地址
               &pim_request,              // 父请求指针
               dramreq_type               // kGEMV
           ));
       }
   }

4. 最终 pimrequest.command_queue:
   [
     PIMCommand { pimcmd_type=kMAC, op_type=kSrc, addr_vec={...} },
     PIMCommand { pimcmd_type=kMAC, op_type=kSrc, addr_vec={...} },
     ... (16 个 kMAC 命令)
   ]
```

### 2.4 地址生成

文件：`src/dram/memory_object.cpp` + `src/dram/mmap_controller.cpp`

```
MemoryObject::getAddrVec(bundle_idx)
  │
  ├─ target_address = address + bundle_idx × granul
  │   (如 0x1000 + 5 × 256 = 0x1500)
  │
  └─ mmap_controller->getAddrVec(target_address)
       │
       └─ addrToVec(target_address)
            │
            ├─ granul 对齐: addr /= granul
            ├─ 逐层取模:
            │   cube    = addr % num_cube    (如 0)
            │   channel = addr % num_channel (如 3)
            │   rank    = addr % num_rank    (如 0)
            │   bg      = addr % num_bankgroup (如 1)
            │   bank    = addr % num_bank    (如 2)
            │   col     = addr % num_col     (如 47)
            │   row     = addr / num_col     (如 12)
            │
            └─ 转换为 Ramulator 地址向量:
                 { channel/2, channel%2, rank, bg, bank, row, col }
                 = { 1, 1, 0, 0, 0, 12, 47 }
```

---

## 第 3 步：发送到 Frontend（PIMController）

### 3.1 SendRequest

```cpp
void DRAMInterface::SendRequest(PIMRequest& pimrequest) {
    frontend->send(pimrequest);
}
```

### 3.2 PIMController::send

文件：`src/dram/ramulator2/src/frontend/impl/PIM_controller.cpp`

```cpp
void send(llm_system::PIMRequest& pimrequest) override {
    // 记录起始时间
    pimrequest.start = m_memory_system->m_clk;  // 如 clk = 1000
    pimrequest.end = m_memory_system->m_clk;
    
    // 遍历每个 PIMCommand
    for (auto& cmd : pimrequest.command_queue) {
        // 1. PIM 命令类型 → DRAM Request 类型
        auto type = get_type(cmd.pimcmd_type);
        //    kMAC → Request::Type::AllRead (= 2)
        
        // 2. 创建 Ramulator Request 并发送到 MemorySystem
        m_memory_system->send({
            cmd.addr_vec,          // 物理地址向量
            get_type(cmd.pimcmd_type), // AllRead
            cmd,                   // PIMCommand (携带回调信息)
            &set_end_time          // 完成回调
        });
    }
}
```

### 3.3 PIMCommand → DRAM Request 映射

```cpp
int get_type(PIMCommandType type) {
    switch (type) {
        case kMAC:      // GEMV 翻译的 kMAC
        case kAdd:
        case kSub:
        case kMult:
        case kDRAM2RF:
            return Request::Type::AllRead;   // → 2 (PIM 读-计算)
        case kRF2DRAM:
            return Request::Type::AllWrite;  // → 3 (PIM 写回)
        case kRead:
            return Request::Type::Read;      // → 0 (普通读)
        case kWrite:
            return Request::Type::Write;     // → 1 (普通写)
    }
}
```

---

## 第 4 步：MemorySystem 路由

### 4.1 MemorySystem::send

文件：`src/dram/ramulator2/src/memory_system/impl/PIM_DRAM_system.cpp`

```cpp
bool send(Request req) override {
    // 1. 地址映射：将 addr_vec 转换为具体 channel
    m_addr_mapper->apply(req);
    int channel_id = req.addr_vec[0];  // 如 channel_id = 1
    
    // 2. 发送到对应 channel 的 DRAM Controller
    bool is_success = m_controllers[channel_id]->send(req);
    return is_success;
}
```

### 4.2 PIMDRAMController::send

文件：`src/dram/ramulator2/src/dram_controller/impl/PIM_dram_controller.cpp`

```cpp
bool send(Request& req) override {
    // 1. 翻译最终命令类型
    req.final_command = m_dram->m_request_translations(req.type_id);
    //    AllRead → 实际的 DRAM 命令序列
    
    // 2. 入队到对应的 Buffer
    req.arrive = m_clk;  // 记录到达时间
    
    if (req.type_id == Request::Type::Read) {
        is_success = m_read_buffer.enqueue(req);
    } else if (req.type_id == Request::Type::Write) {
        is_success = m_write_buffer.enqueue(req);
    } else if (req.type_id == Request::Type::AllRead) {
        is_success = m_PIM_buffer.enqueue(req);   // ← kMAC 进 PIM 专用 buffer
    } else if (req.type_id == Request::Type::AllWrite) {
        is_success = m_PIM_buffer.enqueue(req);
    }
}
```

---

## 第 5 步：逐周期时序模拟

### 5.1 DRAMInterface::run()

```cpp
void DRAMInterface::run() {
    for (uint64_t i = 0;; i++) {
        memory_system->tick();       // 推进一个 DRAM 周期
        if (memory_system->is_finished()) {
            break;                   // 所有 buffer 为空时退出
        }
    }
}
```

### 5.2 MemorySystem::tick() 每周期执行

```
m_clk++                              // 周期 +1
m_dram->tick()                       // DRAM 状态更新
setPIMmode(true)                     // 设置 PIM 模式
for each controller:
    controller->tick()               // 每个 channel 控制器 tick
```

### 5.3 PIMDRAMController::tick() 每周期

```
1. serve_completed_reads()
   │  检查 pending 队列第一条
   │  if req.depart <= m_clk:
   │      调用 req.callback(req)    ───→ set_end_time(req)
   │      pending.pop_front()
   │
2. m_refresh->tick()                 // 刷新管理
   │
3. schedule_request(req_it, buffer)  // 调度请求
   │  ├─ 优先: m_active_buffer (已激活的行)
   │  ├─ 其次: m_PIM_buffer (PIM 请求) ← kMAC 在这里
   │  ├─ 再次: m_priority_buffer (刷新等)
   │  └─ 最后: read/write buffer
   │
4. plugin->update()                  // 插件更新（Hydra 等）
   │
5. if request_found:
      m_dram->issue_command(cmd, addr_vec)  // 发出 DRAM 命令
      │  如: ACT → 激活行
      │      RD  → 读数据
      │      PRE → 预充电
      │
      if cmd == final_command:       // 最后一个命令
          if type == AllRead:
              req.depart = m_clk + read_latency
              pending.push_back(req) // 进入等待完成队列
          buffer->remove(req_it)     // 从 buffer 移除
      else:
          if cmd 是行打开命令:
              m_active_buffer.enqueue(req)  // 移到 active buffer
```

### 5.4 回调触发

```cpp
// 当 pending 队列中的请求 depart 时间到达:
void serve_completed_reads() {
    auto& req = pending[0];
    if (req.depart <= m_clk) {
        if (req.callback) {
            req.callback(req);  // 调用 set_end_time
        }
        pending.pop_front();
    }
}
```

### 5.5 set_end_time 回调

文件：`src/dram/ramulator2/src/frontend/impl/PIM_controller.cpp`

```cpp
void set_end_time(Request& req) {
    // 记录完成时间到 PIMRequest
    req.pimcmd.request->end = req.depart;
    // pimrequest.end = max(所有命令的 depart)
    
    // 累计各类型 DRAM 命令计数
    for (int i = 0; i < (int)llm_system::DRAMCommandType::kMAX; i++) {
        req.pimcmd.request->issued_dram_cmd[i] += req.issued_dram_cmd[i];
        // 如: issued_dram_cmd[ACT] += 1
        //      issued_dram_cmd[RD]  += 1
        //      issued_dram_cmd[PRE] += 1
    }
}
```

---

## 第 6 步：统计收集

### 6.1 updateStatus

```cpp
void DRAMInterface::updateStatus(const PIMRequest& pimrequest) {
    // 执行时间 (DRAM 周期)
    cycle_t duration = pimrequest.end - pimrequest.start;
    // 如: 1200 - 1000 = 200 cycles
    
    // 转换为纳秒并累计
    time += (duration * memory_scale_factor);
    exec_status.memory_duration += (duration * memory_scale_factor);
    
    // 按 HBM3 顺序收集各命令计数:
    exec_status.act_count   += pimrequest.issued_dram_cmd[0];  // ACT
    exec_status.read_count  += pimrequest.issued_dram_cmd[4];  // RD
    exec_status.write_count += pimrequest.issued_dram_cmd[5];  // WR
    exec_status.all_act_count  += pimrequest.issued_dram_cmd[6];
    exec_status.all_read_count += pimrequest.issued_dram_cmd[7];
    exec_status.all_write_count+= pimrequest.issued_dram_cmd[8];
    exec_status.ref_count   += pimrequest.issued_dram_cmd[9];  // REF
}
```

### 6.2 返回到上层

```
updateStatus() 执行完毕
  → HandleRequest() 返回
    → Device::run_ramulator() 返回
      → issueRamulator() 返回 ExecStatus
        → AttentionGenExecutionPIM() 累加 ExecStatus
          → 回到 Module / Executor 层级
```

---

## 完整时间线示例

假设一个 16-bundle 的 GEMV 请求：

```
Cycle 1000: DRAMInterface::HandleRequest() 开始
            ├─ GeneratePIMCommand() → 生成 16 个 kMAC 命令
            └─ SendRequest() → PIMController::send()
                └─ 16 个 AllRead Request → PIM buffer

Cycle 1001: tick() 开始
            ├─ schedule_request() → 从 PIM buffer 取出第一个 AllRead
            ├─ issue_command(ACT, addr)  → 激活目标行
            └─ Request 移到 active_buffer

Cycle 1002: tick()
            ├─ schedule_request() → active_buffer 中的请求
            ├─ issue_command(RD, addr)   → 发出读命令
            └─ Request 进入 pending（等待 tRL）

Cycle 1003: tick()
            ├─ m_PIM_buffer 中下一个 AllRead → ACT
            ...

Cycle 1005: serve_completed_reads()
            ├─ 第一个请求 depart 时间到达
            ├─ callback(req) → set_end_time()
            │     └─ pimrequest.end = max(..., 1005)
            └─ pending.pop_front()

... (继续处理剩余 15 个请求)

Cycle 1200: 最后一个请求完成
            └─ pimrequest.end = 1200

Cycle 1200: is_finished() → true → run() 退出

            updateStatus():
            ├─ duration = 1200 - 1000 = 200 cycles
            ├─ issued_dram_cmd: {ACT:16, RD:16, PRE:16, ...}
            └─ time += 200 * memory_scale_factor
```

---

## 多层 Buffer 优先级

PIMDRAMController 中的请求调度优先级：

```
优先级从高到低:
1. m_active_buffer    — 已激活行，避免重复 ACT
2. m_PIM_buffer       — PIM 请求（AllRead/AllWrite），kMAC 在这里
3. m_priority_buffer  — 刷新等维护操作（最高优先但较少触发）
4. m_read_buffer      — 普通读
5. m_write_buffer     — 普通写（有低水位/高水位切换逻辑）
```

---

## 关键文件索引

| 步骤 | 文件 | 函数 |
|------|------|------|
| 请求生成 | `src/hardware/attention_gen_impl.cpp` | `AttentionGenExecutionPIM()` |
| DRAMRequest 构造 | `src/hardware/layer_impl.cpp` | `issueRamulator()` |
| 入口 | `src/hardware/device.cpp` | `Device::run_ramulator()` |
| 主控 | `src/dram/dram_interface.cpp` | `HandleRequest()` |
| PIM 翻译 | `src/dram/dram_interface.cpp` | `GeneratePIMCommand()` |
| GEMV Kernel | `src/dram/pimkernel/GEMV.cpp` | `GEMV()` |
| 地址生成 | `src/dram/memory_object.cpp` | `MemoryObject::getAddrVec()` |
| 地址映射 | `src/dram/mmap_controller.cpp` | `MMapController::getAddrVec()` |
| 发送 | `src/dram/dram_interface.cpp` | `SendRequest()` |
| PIMController | `ramulator2/src/frontend/impl/PIM_controller.cpp` | `send()` / `get_type()` |
| MemorySystem | `ramulator2/src/memory_system/impl/PIM_DRAM_system.cpp` | `send()` / `tick()` |
| DRAM Controller | `ramulator2/src/dram_controller/impl/PIM_dram_controller.cpp` | `send()` / `tick()` / `schedule_request()` |
| 回调 | `ramulator2/src/frontend/impl/PIM_controller.cpp` | `set_end_time()` |
| 统计 | `src/dram/dram_interface.cpp` | `updateStatus()` |
