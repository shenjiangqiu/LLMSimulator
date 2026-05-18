#!/usr/bin/env python3
"""
Parse LLMSimulator text output → nested JSON with raw .txt saved alongside.

Usage:
    ./build_release/run config.yaml 2>&1 | python3 tools/parse_layer.py -o log/exp.json
    # Produces: log/exp.json  +  log/exp.txt (raw text)

Or pipe directly:
    ./build_release/run config.yaml 2>&1 | python3 tools/parse_layer.py
"""
import sys, re, json, os, argparse
from typing import Optional


def parse_line(line: str) -> Optional[dict]:
    indent = 0
    s = line
    while s.startswith('\t'):
        indent += 1
        s = s[1:]

    m = re.match(r'^(\S+)\s*\|\s*([\d.]+)us\s*\|\s*([\d.]+)\s*-\s*([\d.]+)\s*\|(.*)', s)
    if not m:
        return None
    name, dur, start, end, rest = m.groups()
    proc = None
    if ', GPU' in rest or rest.rstrip().endswith('GPU'):
        proc = 'GPU'
    elif ', PIM' in rest or rest.rstrip().endswith('PIM'):
        proc = 'PIM'
    elif ', Logic' in rest:
        proc = 'Logic'
    return {
        'name': name.strip(),
        'duration_us': float(dur),
        'start_us': float(start),
        'end_us': float(end),
        'processor': proc,
        'indent': indent,
        'children': [],
    }


def build_tree(ops: list) -> dict:
    root = ops[0].copy()
    stack = [root]
    for op in ops[1:]:
        node = op.copy()
        while stack and stack[-1]['indent'] >= op['indent']:
            stack.pop()
        if stack:
            stack[-1]['children'].append(node)
        stack.append(node)
    return root


def classify_stage(node: dict, result: dict) -> None:
    result.setdefault('prefill_us', 0)
    result.setdefault('decode_us', 0)
    name = node['name']
    dur = node['duration_us']
    if 'Sum' in name:
        result['prefill_us'] += dur
        node['stage'] = 'prefill'
    elif 'Gen' in name:
        result['decode_us'] += dur
        node['stage'] = 'decode'
    elif name in ('AttentionSplit', 'AttentionMerge'):
        node['stage'] = 'overhead'
    else:
        st = {}
        for c in node.get('children', []):
            classify_stage(c, st)
        if st.get('prefill_us', 0) > st.get('decode_us', 0):
            result['prefill_us'] += dur
            node['stage'] = 'prefill'
        elif st.get('decode_us', 0) > 0:
            result['decode_us'] += dur
            node['stage'] = 'decode'
        else:
            node['stage'] = 'overhead'


def clean_children(node: dict) -> dict:
    out = {
        'name': node['name'],
        'duration_us': node['duration_us'],
        'start_us': node['start_us'],
        'end_us': node['end_us'],
    }
    if node.get('processor'):
        out['processor'] = node['processor']
    if node.get('stage'):
        out['stage'] = node['stage']
    if node.get('children'):
        out['children'] = [clean_children(c) for c in node['children']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', help='JSON output path (raw .txt saved alongside)')
    args = ap.parse_args()

    raw_text = sys.stdin.read()
    lines = raw_text.splitlines()

    total_us = 0
    for line in lines:
        m = re.search(r'LLM\s*\|\s*([\d.]+)us', line)
        if m:
            total_us = float(m.group(1))
            break

    layer_ops = []
    in_layer = False
    layer_indent = None
    for line in lines:
        p = parse_line(line)
        if not p:
            continue
        if re.match(r'(decoder_\d+|MoE_decoder_\d+)$', p['name']):
            in_layer = True
            layer_indent = p['indent']
            layer_ops.append(p)
        elif in_layer and p['indent'] > layer_indent:
            layer_ops.append(p)
        elif in_layer and p['indent'] <= layer_indent:
            break

    if not layer_ops:
        result = {"error": "no decoder layer found"}
    else:
        root = layer_ops[0]
        tree = build_tree(layer_ops)
        stages = {'prefill_us': 0, 'decode_us': 0}
        classify_stage(tree, stages)
        result = {
            "total_time_us": total_us,
            "first_layer": root['name'],
            "layer_duration_us": root['duration_us'],
            "prefill_us": stages['prefill_us'],
            "decode_us": stages['decode_us'],
            "operations": clean_children(tree),
        }

    json_out = json.dumps(result, indent=2, default=str)

    if args.output:
        json_path = args.output
        txt_path = os.path.splitext(args.output)[0] + '.txt'
        with open(json_path, 'w') as f:
            f.write(json_out)
            f.write('\n')
        with open(txt_path, 'w') as f:
            f.write(raw_text)
        print(f"Wrote {json_path} + {txt_path}")
    else:
        print(json_out)


if __name__ == '__main__':
    main()
