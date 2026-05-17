#!/usr/bin/env python3
"""
LLMSimulator Result Viewer

Parses the hierarchical output from LLMSimulator and generates an
interactive HTML page with collapsible tree, color-coded processor types,
sorting, and filtering.

Usage:
    # From a file:
    python tools/view_result.py output.txt

    # Pipe directly:
    ./build/run config.yaml 2>&1 | python tools/view_result.py

    # From file, specify output HTML:
    python tools/view_result.py output.txt -o report.html
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_line(line):
    """Parse a single line of LLMSimulator output into structured data."""
    # Strip leading tabs to determine depth
    stripped = line.lstrip("\t")
    depth = len(line) - len(stripped)

    # Pattern: name | time | start - end | tensor info | energy info | util info [, ProcessorType]
    # Examples:
    # LLM | 9688.6us  | 0 - 9688.6 | tensor(...) -> tensor(...) | ACT: XmJ ...
    # attn_q_down_proj | 1.3806us  | 0.008064 - 1.3887 | ... | Op/B 7.9748, GPU

    # Only parse lines matching the tree-node pattern: name | numberus | ...
    if not re.match(r'\S+\s*\|\s*[\d.]+us', stripped):
        return None

    result = {"depth": depth, "raw": stripped, "children": []}

    # Extract name (before first |)
    parts = stripped.split("|")
    if len(parts) < 1:
        return None

    result["name"] = parts[0].strip()

    # Extract time
    time_match = re.search(r"([\d.]+)us", stripped)
    if time_match:
        result["time_us"] = float(time_match.group(1))

    # Extract start/end
    range_match = re.search(r"\|\s*([\d.]+)\s*-\s*([\d.]+)", stripped)
    if range_match:
        result["start_us"] = float(range_match.group(1))
        result["end_us"] = float(range_match.group(2))

    # Extract Op/B
    opb_match = re.search(r"Op/B\s+([\d.]+)", stripped)
    if opb_match:
        result["opb"] = float(opb_match.group(1))

    # Extract processor type
    proc_match = re.search(r",\s*(GPU|PIM|Logic)$", stripped)
    if proc_match:
        result["processor"] = proc_match.group(1)
    else:
        result["processor"] = None  # aggregate node

    # Extract energies
    energy_map = {
        "ACT": r"ACT:\s*([\d.]+)mJ",
        "RD": r"RD:\s*([\d.]+)mJ",
        "WR": r"WR:\s*([\d.]+)mJ",
        "all_ACT": r"all_ACT:\s*([\d.]+)mJ",
        "all_RD": r"all_RD:\s*([\d.]+)mJ",
        "all_WR": r"all_WR:\s*([\d.]+)mJ",
        "MAC": r"MAC:\s*([\d.]+)mJ",
        "Total": r"Total:\s*([\d.]+)mJ",
    }
    result["energy"] = {}
    for key, pattern in energy_map.items():
        m = re.search(pattern, stripped)
        if m:
            result["energy"][key] = float(m.group(1))

    # Extract utilization
    compute_match = re.search(r"compute util:\s*([\d.]+)", stripped)
    memory_match = re.search(r"memory util:\s*([\d.]+)", stripped)
    if compute_match:
        result["compute_util"] = float(compute_match.group(1))
    if memory_match:
        result["memory_util"] = float(memory_match.group(1))

    return result


def build_tree(lines):
    """Build a tree from parsed lines using depth-based nesting."""
    root = None
    stack = []  # (depth, node)

    for line in lines:
        node = parse_line(line)
        if node is None:
            continue

        # Find the right parent
        while stack and stack[-1][0] >= node["depth"]:
            stack.pop()

        if not stack:
            if root is None:
                root = node
            # else: skip extra top-level items (like the "Using synthesis trace" line)
        else:
            stack[-1][1]["children"].append(node)

        stack.append((node["depth"], node))

    return root


def compute_stats(node):
    """Compute aggregate statistics for a node."""
    stats = {
        "total_time": node.get("time_us", 0),
        "total_energy": node.get("energy", {}).get("Total", 0),
        "processor_types": set(),
        "child_count": 0,
        "leaf_count": 0,
    }

    if node.get("processor"):
        stats["processor_types"].add(node["processor"])

    for child in node.get("children", []):
        child_stats = compute_stats(child)
        stats["processor_types"].update(child_stats["processor_types"])
        stats["child_count"] += 1 + child_stats["child_count"]
        stats["leaf_count"] += child_stats["leaf_count"] or (
            1 if not child.get("children") else 0
        )

    if not node.get("children"):
        stats["leaf_count"] = 1

    node["_stats"] = stats
    return stats


PROCESSOR_COLORS = {
    "GPU": "#76b900",  # NVIDIA green
    "PIM": "#e85d47",  # Red-orange
    "Logic": "#4a90d9",  # Blue
    None: "#888888",  # Gray for aggregate
}

PROCESSOR_BG = {
    "GPU": "#f0faf0",
    "PIM": "#fff5f3",
    "Logic": "#f0f5fc",
    None: "#fafafa",
}


def generate_html(root, title="LLMSimulator Result"):
    """Generate interactive HTML page."""
    root_json = json.dumps(root, indent=2, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: #1a1a2e; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .controls {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.header button, .header select {{ padding: 6px 14px; border: 1px solid #444; background: #2a2a4e; color: #ddd; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.header button:hover {{ background: #3a3a6e; }}
.summary {{ display: flex; gap: 16px; padding: 12px 24px; background: white; border-bottom: 1px solid #e0e0e0; flex-wrap: wrap; font-size: 13px; }}
.summary .stat {{ display: flex; gap: 6px; align-items: center; }}
.summary .stat .label {{ color: #888; }}
.summary .stat .value {{ font-weight: 600; }}
.stat-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; width: 200px; }}
.stat-bar .seg {{ height: 100%; }}
.tree {{ padding: 16px 24px; max-width: 1400px; }}
.node {{ margin-left: 20px; }}
.node-header {{ display: flex; align-items: center; gap: 8px; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 13px; line-height: 1.6; }}
.node-header:hover {{ background: #e8e8e8; }}
.toggle {{ width: 16px; font-size: 10px; color: #999; flex-shrink: 0; text-align: center; user-select: none; }}
.toggle.collapsed {{ }}
.node-name {{ font-weight: 500; white-space: nowrap; }}
.node-time {{ color: #666; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.node-range {{ color: #aaa; font-size: 11px; white-space: nowrap; }}
.node-opb {{ white-space: nowrap; font-weight: 500; }}
.node-energy {{ color: #888; font-size: 11px; white-space: nowrap; }}
.processor-tag {{ padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; color: white; white-space: nowrap; }}
.children {{ }}
.children.collapsed {{ display: none; }}
.hidden {{ display: none !important; }}
.time-bar {{ display: inline-block; height: 14px; border-radius: 3px; vertical-align: middle; opacity: 0.5; }}
.highlight {{ background: #fff3cd !important; }}
#stats-panel {{ padding: 16px 24px; background: white; border-top: 1px solid #e0e0e0; font-size: 13px; display: none; }}
#stats-panel.show {{ display: block; }}
.legend {{ display: flex; gap: 16px; align-items: center; }}
.legend-item {{ display: flex; gap: 4px; align-items: center; font-size: 12px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>

<div class="header">
    <h1>LLMSimulator Result Viewer</h1>
    <div class="controls">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
        <button onclick="expandToDepth(3)">Depth 3</button>
        <button onclick="resetFilter()">Reset Filter</button>
        <select id="filterProc" onchange="applyFilter()">
            <option value="all">All Processors</option>
            <option value="GPU">GPU</option>
            <option value="PIM">PIM</option>
            <option value="Logic">Logic</option>
            <option value="aggregate">Aggregate (no proc)</option>
        </select>
        <select id="sortBy" onchange="applySort()">
            <option value="tree">Tree Order</option>
            <option value="time">Sort by Time ↓</option>
            <option value="opb">Sort by Op/B ↓</option>
            <option value="energy">Sort by Energy ↓</option>
        </select>
        <button onclick="toggleStats()">Statistics</button>
    </div>
</div>

<div class="summary" id="summary"></div>

<div id="stats-panel"></div>

<div class="tree" id="tree"></div>

<div class="legend" style="padding: 8px 24px; border-top: 1px solid #e0e0e0;">
    <span style="color:#888;font-size:12px;">Processor:</span>
    <div class="legend-item"><div class="legend-dot" style="background:#76b900"></div>GPU</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e85d47"></div>PIM</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4a90d9"></div>Logic</div>
    <div class="legend-item"><div class="legend-dot" style="background:#aaa"></div>Aggregate</div>
</div>

<script>
const DATA = {root_json};

function buildNode(node, depth) {{
    if (!node) return '';

    const hasChildren = node.children && node.children.length > 0;
    const proc = node.processor;
    const color = {json.dumps(PROCESSOR_COLORS)};
    const bg = {json.dumps(PROCESSOR_BG)};
    const procColor = color[proc] || color["null"];
    const procBg = bg[proc] || bg["null"];

    let tagHtml = '';
    if (proc) {{
        tagHtml = `<span class="processor-tag" style="background:${{procColor}}">${{proc}}</span>`;
    }}

    const time = node.time_us ? node.time_us.toFixed(2) + 'us' : '';
    const range = node.start_us != null ? `${{node.start_us.toFixed(2)}} - ${{node.end_us.toFixed(2)}}` : '';
    const opb = node.opb != null ? `Op/B ${{node.opb.toFixed(2)}}` : '';
    const totalEnergy = node.energy && node.energy.Total ? `Tot ${{node.energy.Total.toFixed(2)}}mJ` : '';

    const maxTime = DATA.time_us || 1;
    const barWidth = node.time_us ? (node.time_us / maxTime * 200) : 0;

    let header = `<div class="node-header" data-proc="${{proc || 'aggregate'}}" data-time="${{node.time_us || 0}}" data-opb="${{node.opb || 0}}" data-energy="${{node.energy?.Total || 0}}" style="background:${{procBg}}">
        <span class="toggle" onclick="toggleNode(this)">${{hasChildren ? '▼' : '·'}}</span>
        <span class="time-bar" style="width:${{barWidth.toFixed(0)}}px; background:${{procColor}}"></span>
        <span class="node-name">${{node.name}}</span>
        <span class="node-time">${{time}}</span>
        <span class="node-range">${{range}}</span>
        ${{tagHtml}}
        <span class="node-opb">${{opb}}</span>
        <span class="node-energy">${{totalEnergy}}</span>
    </div>`;

    let childrenHtml = '';
    if (hasChildren) {{
        childrenHtml = '<div class="children">' + node.children.map(c => buildNode(c, depth + 1)).join('') + '</div>';
    }}

    return '<div class="node">' + header + childrenHtml + '</div>';
}}

function toggleNode(el) {{
    const node = el.closest('.node');
    const children = node.querySelector(':scope > .children');
    if (children) {{
        children.classList.toggle('collapsed');
        el.textContent = children.classList.contains('collapsed') ? '▶' : '▼';
    }}
}}

function expandAll() {{
    document.querySelectorAll('.children').forEach(c => c.classList.remove('collapsed'));
    document.querySelectorAll('.toggle').forEach(t => {{
        if (t.closest('.node').querySelector(':scope > .children')) t.textContent = '▼';
    }});
}}

function collapseAll() {{
    document.querySelectorAll('.children').forEach(c => c.classList.add('collapsed'));
    document.querySelectorAll('.toggle').forEach(t => {{
        if (t.closest('.node').querySelector(':scope > .children')) t.textContent = '▶';
    }});
}}

function expandToDepth(maxDepth) {{
    collapseAll();
    document.querySelectorAll('.node').forEach(node => {{
        const depth = getDepth(node);
        if (depth <= maxDepth) {{
            const children = node.querySelector(':scope > .children');
            if (children) {{
                children.classList.remove('collapsed');
                const toggle = node.querySelector(':scope > .node-header > .toggle');
                if (toggle) toggle.textContent = '▼';
            }}
        }}
    }});
}}

function getDepth(node) {{
    let depth = 0;
    let parent = node.parentElement;
    while (parent) {{
        if (parent.classList.contains('node')) depth++;
        parent = parent.parentElement;
    }}
    return depth;
}}

function applyFilter() {{
    const filter = document.getElementById('filterProc').value;
    document.querySelectorAll('.node-header').forEach(header => {{
        const proc = header.dataset.proc;
        if (filter === 'all') {{
            header.closest('.node').classList.remove('hidden');
        }} else if (filter === 'aggregate') {{
            header.closest('.node').classList.toggle('hidden', proc !== 'aggregate');
        }} else {{
            header.closest('.node').classList.toggle('hidden', proc !== filter);
        }}
    }});
}}

function resetFilter() {{
    document.getElementById('filterProc').value = 'all';
    document.getElementById('sortBy').value = 'tree';
    document.querySelectorAll('.node').forEach(n => n.classList.remove('hidden'));
    // Rebuild to reset sort
    document.getElementById('tree').innerHTML = buildNode(DATA, 0);
    collapseAll();
    expandToDepth(3);
    computeSummary();
}}

function applySort() {{
    const sortBy = document.getElementById('sortBy').value;
    if (sortBy === 'tree') {{
        document.getElementById('tree').innerHTML = buildNode(DATA, 0);
        collapseAll();
        expandToDepth(3);
        return;
    }}

    const allNodes = [];
    document.querySelectorAll('.node-header').forEach(h => {{
        allNodes.push({{
            el: h.closest('.node'),
            time: parseFloat(h.dataset.time) || 0,
            opb: parseFloat(h.dataset.opb) || 0,
            energy: parseFloat(h.dataset.energy) || 0,
        }});
    }});

    const key = sortBy;
    allNodes.sort((a, b) => b[key] - a[key]);

    const tree = document.getElementById('tree');
    allNodes.forEach(n => tree.appendChild(n.el));
}}

function computeSummary() {{
    let totalTime = DATA.time_us || 0;
    let totalEnergy = DATA.energy?.Total || 0;

    // Count processors
    let counts = {{ GPU: 0, PIM: 0, Logic: 0, aggregate: 0 }};
    function countNodes(node) {{
        if (node.processor) counts[node.processor] = (counts[node.processor] || 0) + 1;
        else counts.aggregate++;
        (node.children || []).forEach(countNodes);
    }}
    countNodes(DATA);

    // Find max time for bar scaling
    let maxChildTime = 0;
    function findMax(node) {{
        if (node.time_us > maxChildTime) maxChildTime = node.time_us;
        (node.children || []).forEach(findMax);
    }}
    findMax(DATA);

    document.getElementById('summary').innerHTML = `
        <div class="stat"><span class="label">Total Time:</span><span class="value">${{totalTime.toFixed(1)}} us</span></div>
        <div class="stat"><span class="label">Total Energy:</span><span class="value">${{totalEnergy.toFixed(2)}} mJ</span></div>
        <div class="stat"><span class="label">GPU:</span><span class="value">${{counts.GPU || 0}} nodes</span></div>
        <div class="stat"><span class="label">PIM:</span><span class="value">${{counts.PIM || 0}} nodes</span></div>
        <div class="stat"><span class="label">Logic:</span><span class="value">${{counts.Logic || 0}} nodes</span></div>
    `;
}}

function toggleStats() {{
    const panel = document.getElementById('stats-panel');
    if (panel.classList.contains('show')) {{
        panel.classList.remove('show');
        return;
    }}

    // Build statistics
    const flatList = [];
    function flatten(node, path) {{
        const entry = {{
            name: node.name,
            path: path,
            time: node.time_us || 0,
            opb: node.opb || 0,
            energy: node.energy?.Total || 0,
            processor: node.processor || '-',
        }};
        flatList.push(entry);
        (node.children || []).forEach(c => flatten(c, path + ' > ' + node.name));
    }}
    flatten(DATA, '');

    // Top 10 by time
    const byTime = [...flatList].sort((a, b) => b.time - a.time).slice(0, 10);
    const byOpb = [...flatList].sort((a, b) => b.opb - a.opb).slice(0, 10);
    const byEnergy = [...flatList].sort((a, b) => b.energy - a.energy).slice(0, 10);

    panel.innerHTML = `
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="flex:1;min-width:300px;">
                <h3 style="margin-bottom:8px;">Top 10 by Time</h3>
                <table style="width:100%;border-collapse:collapse;font-size:12px;">
                    <tr style="color:#888;"><th style="text-align:left;">Name</th><th>Time</th><th>Op/B</th><th>Proc</th></tr>
                    ${{byTime.map(r => `<tr><td>${{r.name}}</td><td>${{r.time.toFixed(2)}}us</td><td>${{r.opb.toFixed(1)}}</td><td>${{r.processor}}</td></tr>`).join('')}}
                </table>
            </div>
            <div style="flex:1;min-width:300px;">
                <h3 style="margin-bottom:8px;">Top 10 by Energy</h3>
                <table style="width:100%;border-collapse:collapse;font-size:12px;">
                    <tr style="color:#888;"><th style="text-align:left;">Name</th><th>Energy</th><th>Op/B</th><th>Proc</th></tr>
                    ${{byEnergy.map(r => `<tr><td>${{r.name}}</td><td>${{r.energy.toFixed(2)}}mJ</td><td>${{r.opb.toFixed(1)}}</td><td>${{r.processor}}</td></tr>`).join('')}}
                </table>
            </div>
        </div>
    `;
    panel.classList.add('show');
}}

// Initialize
document.getElementById('tree').innerHTML = buildNode(DATA, 0);
collapseAll();
expandToDepth(3);
computeSummary();
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="LLMSimulator Result Viewer")
    parser.add_argument("input", nargs="?", help="Input file (stdin if not provided)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML file (default: result_view.html)",
    )
    args = parser.parse_args()

    # Read input
    if args.input:
        lines = open(args.input).readlines()
    else:
        lines = sys.stdin.readlines()

    # Parse and build tree
    root = build_tree(lines)
    if root is None:
        print("Error: Could not parse any valid output lines.", file=sys.stderr)
        sys.exit(1)

    compute_stats(root)

    # Generate HTML
    html = generate_html(root)

    # Write output
    output_path = args.output or "result_view.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated: {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    main()
