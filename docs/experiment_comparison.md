# Experiment Comparison Report

| Experiment | Total(us) | Layer(us) | Gen(us) | Proc | Q@K(us) | per_seq | S@V(us) | qk_rb | qk_pe | sv_rb | sv_pe | pim_rb | pim_pe |
|------------|-----------|-----------|---------|------|---------|---------|---------|-------|-------|-------|-------|--------|--------|
| exp1_fp16_gpu                |      5623 |       176 |      95 | GPU  |      97 |     6.0 |       0 |     0 |     0 |     0 |     0 |      0 |      0 |
| exp2_fp16_pim                |      6788 |       212 |     132 | PIM  |      66 |     4.1 |      66 |    66 |    33 |    66 |    33 |    131 |     66 |
| exp3_2bit_gpu                |      2769 |        87 |      48 | GPU  |      48 |     3.0 |       0 |     0 |     0 |     0 |     0 |      0 |      0 |
| exp4_2bit_hybrid             |      3460 |       108 |      67 | PIM  |      33 |     2.1 |      34 |    33 |    17 |     0 |     0 |     33 |     17 |
| exp5_2bit_allpim             |      3443 |       108 |      66 | PIM  |      33 |     2.1 |      33 |    33 |    17 |    33 |    17 |     66 |     33 |
| exp6_2bit_dequant_pim        |      5555 |       174 |     132 | PIM  |      66 |     4.1 |      66 |    33 |    50 |    33 |    50 |     66 |     99 |

## PIM Cycle Verification (exp2 FP16 PIM)

- per-GEMV: rb=128.0ns pe=65.1ns
- per-step (×4grp ×8heads ×2dir ×16seqs): rb=131.1us pe=66.6us
- Reported: qk_rb=66us qk_pe=33us sv_rb=66us sv_pe=33us
- ✓ ratios: qk_rb=1.00x sv_rb=1.00x
