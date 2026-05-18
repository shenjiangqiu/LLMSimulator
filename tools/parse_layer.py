#!/usr/bin/env python3
"""
Parse LLMSimulator output and extract the first decoder layer's timing as JSON.
Usage: ./build/run config.yaml 2>&1 | python3 tools/parse_layer.py > layer_0.json
"""
import sys, re, json

def parse_line(line):
    """Parse a line like:
    \t\tMoE_decoder_0 | 398.22us  | 0 - 398.22 | tensor(16, 4096) -> tensor(16, 4096) | ...
    Returns (indent, name, duration_us, start_us, end_us, rest) or None.
    """
    # Count leading tabs
    indent = 0
    stripped = line
    while stripped.startswith('\t'):
        indent += 1
        stripped = stripped[1:]

    # Match: name | duration | start - end | ...
    m = re.match(r'^(\S+)\s*\|\s*([\d.]+)us\s*\|\s*([\d.]+)\s*-\s*([\d.]+)\s*\|(.*)', stripped)
    if not m:
        return None
    name, dur, start, end, rest = m.groups()
    # Extract processor type
    proc = None
    if 'GPU' in rest:
        proc = 'GPU'
    elif 'PIM' in rest:
        proc = 'PIM'
    elif 'Logic' in rest:
        proc = 'Logic'
    return {
        'indent': indent,
        'name': name.strip(),
        'duration_us': float(dur),
        'start_us': float(start),
        'end_us': float(end),
        'processor': proc,
        'rest': rest.strip()
    }

def find_first_layer(lines):
    """Find lines belonging to the first decoder/moe_decoder layer."""
    in_layer = False
    layer_indent = None
    for line in lines:
        p = parse_line(line)
        if not p:
            continue
        # Detect first decoder layer
        if re.match(r'(MoE_decoder_0|decoder_0)$', p['name']):
            in_layer = True
            layer_indent = p['indent']
            yield p
        elif in_layer and p['indent'] > layer_indent:
            yield p
        elif in_layer and p['indent'] <= layer_indent:
            break  # back to same level or higher → done

def classify_stage(name, duration_us):
    """Classify an operation into prefill/decode."""
    if 'Sum' in name or 'sum' in name.lower():
        return 'prefill'
    if 'Gen' in name or 'gen' in name.lower():
        return 'decode'
    # For parent nodes, estimate from children
    return 'unknown'

def main():
    lines = sys.stdin.readlines()
    
    # Find total time
    total_us = 0
    for line in lines:
        m = re.search(r'LLM\s*\|\s*([\d.]+)us', line)
        if m:
            total_us = float(m.group(1))
            break
    
    layer_ops = list(find_first_layer(lines))
    if not layer_ops:
        print(json.dumps({"error": "no decoder layer found"}, indent=2))
        return
    
    root = layer_ops[0]
    
    # Build tree
    stack = [{'name': root['name'], 'duration_us': root['duration_us'],
              'start_us': root['start_us'], 'end_us': root['end_us'],
              'processor': root['processor'], 'children': [], 'level': root['indent']}]
    
    for op in layer_ops[1:]:
        node = {'name': op['name'], 'duration_us': op['duration_us'],
                'start_us': op['start_us'], 'end_us': op['end_us'],
                'processor': op['processor'], 'children': [], 'level': op['indent']}
        # Pop until we find the parent
        while stack and stack[-1]['level'] >= op['indent']:
            stack.pop()
        if stack:
            stack[-1]['children'].append(node)
        stack.append(node)
    
    # Separate prefill and decode times
    def collect_by_stage(node, stage_times):
        name = node['name']
        dur = node['duration_us']
        if 'Sum' in name or 'sum' in name.lower() or 'AttentionSum' in name:
            stage_times['prefill'] += dur
        elif 'Gen' in name or 'gen' in name.lower() or 'AttentionGen' in name:
            stage_times['decode'] += dur
        elif 'AttentionSplit' in name or 'AttentionMerge' in name:
            # Split/Merge span both stages — count under overhead
            stage_times['overhead'] += dur
        else:
            # For parent nodes, look at children
            st = {'prefill': 0, 'decode': 0, 'overhead': 0}
            for child in node.get('children', []):
                collect_by_stage(child, st)
            if st['prefill'] > 0 and st['decode'] > 0:
                stage_times['prefill'] += st['prefill']
                stage_times['decode'] += st['decode']
                stage_times['overhead'] += st['overhead']
            elif st['prefill'] > 0:
                stage_times['prefill'] += dur
            elif st['decode'] > 0:
                stage_times['decode'] += dur
            else:
                stage_times['overhead'] += dur
    
    stage_times = {'prefill': 0, 'decode': 0, 'overhead': 0}
    collect_by_stage({'name': root['name'], 'duration_us': root['duration_us'],
                       'children': layer_ops[1:] if len(layer_ops) > 1 else []},
                      stage_times)
    
    result = {
        "total_time_us": total_us,
        "first_layer": root['name'],
        "layer_duration_us": root['duration_us'],
        "prefill_us": stage_times['prefill'],
        "decode_us": stage_times['decode'],
        "overhead_us": stage_times['overhead'],
        "operations": layer_ops
    }
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == '__main__':
    main()
