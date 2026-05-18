#!/usr/bin/env python3
"""
Generate comparison report across all experiments with per-stage attention breakdown.
Usage: python3 tools/report.py log/config_exp*.json
"""
import json, sys, os, glob
from collections import OrderedDict

def find_node(tree, name):
    """Recursively find a node by name in a nested dict tree."""
    if isinstance(tree, dict):
        if tree.get('name') == name:
            return tree
        for c in tree.get('children', []):
            r = find_node(c, name)
            if r: return r
    elif isinstance(tree, list):
        for item in tree:
            r = find_node(item, name)
            if r: return r
    return None

def parse_per_stage_timing(txt_path):
    """Extract qk/softmax/score_v/kv_quant from the first AttentionGen line."""
    qk = softmax = score_v = kv_quant = 0.0
    if not os.path.exists(txt_path):
        return qk, softmax, score_v, kv_quant
    with open(txt_path) as f:
        for line in f:
            if 'AttentionGen' in line and 'qk=' in line:
                import re
                m = re.search(r'qk=([\d.]+)us softmax=([\d.]+)us score_v=([\d.]+)us kv_quant=([\d.]+)us', line)
                if m:
                    qk = float(m.group(1))
                    softmax = float(m.group(2))
                    score_v = float(m.group(3))
                    kv_quant = float(m.group(4))
                    break  # first occurrence = per-step value
    return qk, softmax, score_v, kv_quant

def main():
    if len(sys.argv) > 1:
        files = sorted(sys.argv[1:])
    else:
        files = sorted(glob.glob('log/config_exp*.json'))
    if not files:
        print("No experiment JSON files found")
        return

    rows = []
    for f in files:
        exp = os.path.basename(f).replace('config_','').replace('.json','')
        d = json.load(open(f))
        txt = f.replace('.json', '.txt')

        ops = d.get('operations', {})
        gen = find_node(ops, 'AttentionGen')
        gen_dur = gen['duration_us'] if gen else 0
        gen_proc = gen.get('processor', '?') if gen else '?'

        qk, softmax, score_v, kv_quant = parse_per_stage_timing(txt)

        rows.append({
            'exp': exp,
            'total_us': d.get('total_time_us', 0),
            'layer_us': d.get('layer_duration_us', 0),
            'attn_gen_us': gen_dur,
            'attn_proc': gen_proc,
            'qk_us': qk,
            'softmax_us': softmax,
            'score_v_us': score_v,
            'kv_quant_us': kv_quant,
            'prefill_us': d.get('prefill_us', 0),
            'decode_us': d.get('decode_us', 0),
        })

    # Print Markdown table
    print("## Experiment Comparison\n")
    print("| Experiment | Total (us) | Layer (us) | AttnGen (us) | Proc | Q@K (us) | Softmax (us) | Score@V (us) | KV Quant (us) |")
    print("|------------|-----------|-----------|-------------|------|---------|-------------|-------------|---------------|")
    for r in rows:
        print(f"| {r['exp']:23s} | {r['total_us']:9.0f} | {r['layer_us']:9.0f} | {r['attn_gen_us']:11.1f} | {r['attn_proc']:4s} | {r['qk_us']:7.1f} | {r['softmax_us']:11.3f} | {r['score_v_us']:11.1f} | {r['kv_quant_us']:13.3f} |")

    # PIM cycle verification
    print("\n## PIM Cycle Verification (exp2 FP16 PIM)\n")
    verify_pim_cycles(rows)

def verify_pim_cycles(rows):
    """Verify PIM latency against analytical cycle count."""
    # For exp2: M=1, K=128, N=16528, elem=2, 256 banks, 8 KV heads, group_size=4
    # Total elements per GEMV: M*K + K*N + M*N = 128 + 128*16528 + 16528 = 2131072
    # Total bytes = 2131072 * 2 = 4262144
    # Per bank bytes = 4262144 / 256 = 16649
    # Rowbuffer fills = ceil(16649 / 2048) = 9
    # RB fill time per fill = 2048 / 32e9 * 1e9 = 64ns
    # Total RB time = 9 * 64 = 576ns
    # PE time = 16649 / 16 * 0.5 = 520.28ns
    # Pipelined latency = 64 + max(8*64, 520.28) = 64 + 520.28 = 584.28ns
    # Per KV head × attention_group_size=4: 584.28 * 4 = 2337ns = 2.34us per GEMV
    # Q@K + score@V = 2 * 2.34 = 4.68us per sequence
    # With 64 batch × 128 steps, cumulative qk should be approx...
    
    M, K, N = 1, 128, 16528
    elem = 2
    banks = 256
    kv_heads = 8
    group = 4  # num_heads/num_kv_heads = 32/8 = 4
    rb_size = 2048
    dram_bw = 32e9
    pe_width = 16
    pe_cycle = 0.5

    total_elem = M*K + K*N + M*N
    total_bytes = total_elem * elem
    per_bank = total_bytes / banks
    rb_fills = max(1, int(__import__('math').ceil(per_bank / rb_size)))
    rb_fill_one = rb_size / dram_bw * 1e9
    rb_total = rb_fills * rb_fill_one
    pe_total = per_bank / pe_width * pe_cycle

    if per_bank <= rb_size:
        latency = max(per_bank / dram_bw * 1e9, pe_total)
    else:
        latency = rb_fill_one + max((rb_fills-1)*rb_fill_one, pe_total)

    per_gemv = latency * group
    print(f"  GEMV params: M={M}, K={K}, N={N}, elem={elem}B")
    print(f"  Total bytes: {total_bytes:.0f}B, per_bank: {per_bank:.0f}B")
    print(f"  Rowbuffer fills: {rb_fills}, fill_time: {rb_fill_one:.1f}ns each")
    print(f"  RB total: {rb_total:.1f}ns, PE total: {pe_total:.1f}ns")
    print(f"  Pipelined latency per GEMV: {latency:.1f}ns")
    print(f"  Per attention group (×{group}): {per_gemv:.1f}ns = {per_gemv/1000:.2f}us")
    print(f"  Per decode step (Q@K + score@V): {2*per_gemv/1000:.2f}us")
    print(f"  With {kv_heads} KV heads and batch=64: {2*per_gemv*kv_heads/1000:.2f}us per step total")

  # Compare with reported
  for r in rows:
    if 'fp16_pim' in r['exp']:
      reported_qk = r['qk_us']
      print(f"\n  Reported: Q@K={reported_qk:.2f}us (per step, all KV heads × batch)")
      # Analytical per step (all KV heads): per_gemv * kv_heads per direction
      analytical_step = per_gemv / 1000 * kv_heads
      print(f"  Analytical per step: {analytical_step:.2f}us (per GEMV {per_gemv/1000:.2f}us × {kv_heads} KV heads)")
      ratio = reported_qk / analytical_step
      print(f"  Ratio reported/analytical: {ratio:.2f}x")
      if 0.9 < ratio < 1.1:
        print("  ✓ PIM cycle verification PASSED")
      else:
        print("  ⚠ PIM cycle verification — check config")

if __name__ == '__main__':
    main()
