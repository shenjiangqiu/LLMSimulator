# PIM-Attn Paper — TODO Checklist

## Figures to Add

### Figure 1 — KV Cache Memory Growth
Bar chart: Llama-3-8B KV cache memory at context lengths 4K, 8K, 16K, 32K, 128K.
Two bars per length: FP16 (2 bytes/elem) vs 2-bit (0.25 bytes/elem).
Data: `2 × n_layers × d_kv × L × elem_bytes` where `d_kv = 1024, n_layers = 32`.

### Figure 2 — Architecture Comparison
Three side-by-side subfigures:
- (a) GPU-only: KV cache in HBM, data moves to GPU SMs for compute
- (b) Logic-die PIM: PE on logic die, data via TSVs from memory banks
- (c) Near-bank PIM: PE inside each DRAM bank, local row buffer

### Figure 3 — Near-Bank PIM Microarchitecture
Zoomed-in single PIM bank:
`DRAM array → sense amps → row buffer (2KB) → PE (W bytes/cycle MAC) → result reg`
Annotate: rowbuffer_size=2048B, pe_width=16B/cycle, dram_to_rb_bw=32GB/s.

### Figure 4 — System Overview
Data flow diagram with 3 phases:
1. Query Q broadcast (GPU → PIM banks) → per-bank Q@K → partial scores
2. Score reduction → softmax on GPU
3. Score@V computed on GPU (V dequantized to FP16)

### Figure 5 — Execution Timeline
Gantt chart: GPU timeline (QKV proj) || PIM timeline (Q@K: rb_fill + pe_compute) → GPU (softmax) → GPU (Score@V + output proj).

### Figure 6 — Asymmetric Quant Overhead
Grouped bar chart: per-GEMV latency (ns), stacked bars for rowbuffer time and PE time.
Two groups: FP16 PIM (with/without asymmetric quant) and 2-bit PIM (with/without).

## Missing Data

### Prefill Mode Results
- [ ] Fix prefill mode KV cache bug in simulator
- [ ] Run prefill experiments (M > 1) to show asymmetric overhead amortization
- [ ] Add prefill results table to experiment section

### Larger Model Results
- [ ] Run Llama-2-70B (80 layers, d_model=8192) experiments
- [ ] Run Llama-3-405B (126 layers, d_model=16384) experiments
- [ ] Add scalability comparison table

### Throughput Metrics
- [ ] Compute tokens/second for each configuration
- [ ] Replace or supplement latency-only metrics
- [ ] Add throughput vs. batch size plot

## Reference Issues

### BibTeX Cleanup
- [ ] Remove duplicate entries from ref.bib (merged KIVI original + our additions)
- [ ] Verify all citation keys match between tex files and bib file
- [ ] Add missing entries: `duplex`, `newton2021newton`, `mutlu2020processing`
