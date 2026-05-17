# Nearbank PIM Unit - Implementation Summary

## Overview

Added a Nearbank Processing-in-Memory (PIM) simulation unit to the LLMSimulator that models near-bank PIM processing for GEMV operations in attention layers.

## Architecture

### New Files
| File | Description |
|------|-------------|
| `src/dram/nearbank/nearbank_config.h` | Configuration struct for Nearbank PIM parameters |
| `src/dram/nearbank/nearbank_pim.h` | NearbankPIMUnit class declaration |
| `src/dram/nearbank/nearbank_pim.cpp` | NearbankPIMUnit implementation with latency model |
| `src/dram/nearbank/CMakeLists.txt` | Build configuration |
| `eval/test_nearbank.cpp` | Unit tests (25 test cases, all passing) |

### Modified Files
| File | Changes |
|------|---------|
| `src/hardware/hardware_config.h` | Added `NearbankPIMConfig` to `SystemConfig` |
| `src/hardware/attention_gen_impl.cpp` | PIM attention gen uses NearbankPIMUnit when enabled |
| `src/hardware/attention_mixed_impl.cpp` | PIM attention mixed uses NearbankPIMUnit when enabled |
| `src/dram/CMakeLists.txt` | Added `nearbank` subdirectory |
| `config.yaml` | Added `nearbank_pim` configuration section |

### Experiment Config Files
| File | Description |
|------|-------------|
| `config_exp1_fp16_gpu.yaml` | Experiment 1: KV FP16, all GPU |
| `config_exp2_fp16_pim.yaml` | Experiment 2: KV FP16, all PIM |
| `config_exp3_2bit_gpu.yaml` | Experiment 3: KV 2-bit, all GPU |
| `config_exp4_2bit_hybrid.yaml` | Experiment 4: KV 2-bit, Q@K in PIM, softmax+score@V on GPU |

## Latency Model

The NearbankPIMUnit uses a pipelined row-buffer model:

1. **Total data movement**: `total_bytes = (M*K + K*N + M*N) * element_size_bytes`
2. **Work distribution**: `bytes_per_bank = total_bytes / num_pim_banks`
3. **Row buffer processing**: Data loaded into rowbuffers (size configurable), PE computes on data in each rowbuffer
4. **Pipelining**: While PE computes on rowbuffer i, rowbuffer i+1 is being filled from DRAM
5. **Latency**: `latency = first_fill_time + max(remaining_fill_time, total_pe_time)`

### Configurable Parameters

```yaml
nearbank_pim:
  enable_nearbank_model: true    # Enable near-bank PIM modeling
  num_pim_banks: 256             # Total PIM banks
  rowbuffer_size_bytes: 2048     # Row buffer size (bytes)
  pe_width_bytes: 16.0           # PE width (bytes processed per cycle)
  pe_cycle_time_ns: 0.5          # PE cycle time (ns)
  dram_to_rb_bw_per_bank: 3.2e10 # DRAM→rowbuffer bandwidth per bank (B/s)
  enable_softmax_in_pim: false   # Softmax in PIM (vs GPU)
  enable_context_in_pim: true    # score@V GEMV in PIM (vs GPU)
  enable_scoring_in_pim: true    # Q@K^T GEMV in PIM (vs GPU)
```

## Design Decisions

1. **Virtual addresses only**: No real data layout simulation. Operand sizes determine virtual addresses internally. Only byte counts matter for latency calculation.

2. **Single bank modeling**: All banks work in parallel; total latency equals one bank's processing time. Work is evenly distributed.

3. **Original LOGIC die code preserved**: The original logical die PE code is maintained but the nearbank model is used for `ProcessorType::PIM` when enabled.

4. **Configurable precision**: Support for FP16 (2 bytes/element) and 2-bit quantized data via `precision_byte` in simulation config.

5. **Pipeline efficiency**: The model accounts for overlap between rowbuffer fills and PE computation, with the first fill acting as pipeline startup overhead.

## Building and Running

```bash
# Build
cd build && cmake .. && make -j

# Run unit tests
./build/test_nearbank

# Run simulator with default config
./build/run config.yaml

# Run experiments
./build/run config_exp1_fp16_gpu.yaml
./build/run config_exp2_fp16_pim.yaml
./build/run config_exp3_2bit_gpu.yaml
./build/run config_exp4_2bit_hybrid.yaml
```

## Test Results

All 25 unit tests pass:
- Basic FP16 GEMV latency computation
- 2-bit quantized GEMV (half the data, lower latency)
- Different PE configurations (fast/wide/slow)
- Different bank counts (64 to 1024 banks)
- Attention scenarios (short/long/very-long sequences)
- Edge cases (single element, small GEMV, multi-token)
- Rowbuffer pipelining verification
- Config validation (throughput calculation)

## Notes for Future Work

1. The current model assumes ideal load balancing across banks. Bank-level contention is not modeled.
2. Reduction of partial results across banks is not explicitly modeled (assumed to be negligible).
3. The YAML parsing for nearbank_pim config in eval/test.cpp should be added for full configurability from config files.
4. The current model assumes decode mode (M=1). For prefill mode with M>1, the model handles it correctly by scaling total bytes.
5. Energy modeling for PIM operations is not included but can be added as extensions to the ExecStatus structure.
