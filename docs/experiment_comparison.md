# Experiment Comparison Report

| Experiment | Total(us) | Layer(us) | AttnGen(us) | Proc | Q@K(us) | per_seq | Softmax(us) | Score@V(us) | KVQuant(us) | pim_rb(us) | pim_pe(us) |
|------------|-----------|-----------|-------------|------|---------|---------|-------------|-------------|-------------|------------|------------|
| exp1_fp16_gpu                |      5623 |       176 |          95 | GPU  |      97 |     6.0 |       0.000 |           0 |       0.000 |          0 |          0 |
| exp2_fp16_pim                |      6788 |       212 |         132 | PIM  |      66 |     4.1 |       0.034 |          66 |       0.008 |        131 |         66 |
| exp3_2bit_gpu                |      2769 |        87 |          48 | GPU  |      48 |     3.0 |       0.000 |           0 |       0.000 |          0 |          0 |
| exp4_2bit_hybrid             |      3460 |       108 |          67 | PIM  |      33 |     2.1 |       0.008 |          34 |       0.008 |         33 |         17 |
| exp5_2bit_allpim             |      3443 |       108 |          66 | PIM  |      33 |     2.1 |       0.008 |          33 |       0.008 |         66 |         33 |
| exp6_2bit_dequant_pim        |      5555 |       174 |         132 | PIM  |      66 |     4.1 |       0.008 |          66 |       0.008 |         66 |         99 |

## PIM Cycle Verification (exp2 FP16 PIM)

- GEMV: M=1 K=128 N=16528 elem=2B, banks=2048, rb_fills=2
- Analytical per-GEMV: rb=128.0ns pe=65.1ns
- Analytical per-step (×4group ×8heads ×2dir ×16seqs): rb=131.1us pe=66.6us
- Reported: pim_rb=131.1us pim_pe=66.5us
- RB ratio: 1.00x  PE ratio: 1.00x
