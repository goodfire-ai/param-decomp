---
name: graph-e2e
description: End-to-end attribution graph analysis — trace dominant pathways from output to earliest components, fetch autointerp labels, analyze asymmetries, and produce a findings summary
user-invocable: true
---

# End-to-End Attribution Graph Analysis

Perform a full analysis of an optimized attribution graph: start from the output node, identify dominant pathways, trace through attention V/K/Q components, fetch autointerp labels, analyze asymmetries, and trace upstream to the earliest differentiating components. Produce a markdown findings document.

## Usage

`/graph-e2e <port> <graph_id>`

- `port` (required): The port the SPD app backend is running on (e.g. `8000`)
- `graph_id` (required): The graph ID to analyze. Must be an optimized graph.

## Workflow

### Step 1: Load the graph

Use the `/graph` skill to load the graph into a Scribe notebook. This sets up `edges_df`, `tokens`, `ci_vals`, `PORT`, and `graph_data`.

### Step 2: Derive the output node

The output node is encoded in the optimization metadata. Execute:

```python
opt = graph_data['optimization']
loss = opt['loss']
output_node = f"output:{loss['position']}:{loss['label_token']}"
print(f"Output node: {output_node}")
print(f"Target token: '{loss['label_str']}' (id {loss['label_token']}) at position {loss['position']}")
print(f"Prompt: {' '.join(tokens)}")
```

### Step 3: Output attribution breakdown

Find all edges into the output node. Group by source layer and show per-layer totals and top individual edges.

```python
import pandas as pd

out_edges = edges_df[edges_df['tgt'] == output_node].copy()

# Parse source layer from node key "layer:seq:cIdx"
out_edges['src_layer'] = out_edges['src'].str.rsplit(':', n=2).str[0]

layer_summary = out_edges.groupby('src_layer').agg(
    sum_val=('val', 'sum'),
    sum_abs_val=('abs_val', 'sum'),
    n=('val', 'count')
).sort_values('sum_abs_val', ascending=False)

print("=== Output attribution by source layer ===")
print(layer_summary.to_string())
print(f"\n=== Top edges to {output_node} ===")
print(out_edges.nlargest(12, 'abs_val')[['src', 'val']].to_string(index=False))
```

From this table, identify the 2-3 dominant pathways. Typical patterns:
- Final MLP (`3.mlp.down`) — usually the largest contributor
- Penultimate attention (`2.attn.o`) — often an induction-like head
- Final attention (`3.attn.o`) — sometimes suppressive (negative val)

### Step 4: Identify attention O nodes to investigate

From the top edges, pick the dominant `*.attn.o` nodes — these carry information from source tokens to the output position via cross-sequence attention.

```python
dominant_o_nodes = out_edges[out_edges['src'].str.contains('attn.o')].nlargest(3, 'abs_val')['src'].tolist()
print("Dominant O nodes:", dominant_o_nodes)
```

### Step 5: V → O cross-sequence analysis

For each dominant O node, find all incoming cross-sequence edges (from `*.attn.v` nodes at other positions). These tell you which source tokens and V components contribute.

```python
for o_node in dominant_o_nodes:
    v_edges = edges_df[
        (edges_df['tgt'] == o_node) &
        (edges_df['is_cross_seq'] == True) &
        (edges_df['src'].str.contains('attn.v'))
    ].copy()
    v_edges['src_pos'] = v_edges['src'].str.split(':').str[1].astype(int)
    v_edges['src_cidx'] = v_edges['src'].str.split(':').str[2]
    v_edges['src_token'] = v_edges['src_pos'].map(lambda p: tokens[p] if p < len(tokens) else '?')

    print(f"\n=== V → {o_node} (cross-sequence) ===")
    print(v_edges.sort_values('abs_val', ascending=False)[['src', 'val', 'src_token']].to_string(index=False))

    # Group by source position
    pos_summary = v_edges.groupby(['src_pos', 'src_token']).agg(
        sum_val=('val', 'sum'), n=('val', 'count')
    ).sort_values('sum_val')
    print(f"\nBy source position:")
    print(pos_summary.to_string())
```

Identify which V component indices are shared across positions and which are position-unique.

### Step 6: V-side asymmetry analysis

This is often the most revealing step. When multiple source positions contribute the same V components, the asymmetry in their contributions explains *why the model predicts token A instead of token B*.

For each V component active at multiple positions, sum its outgoing attribution to **all** O targets (not just the dominant one).

```python
# Get all V nodes at the relevant positions
relevant_positions = [...]  # fill in the positions of interest (e.g. the two variable positions)

v_nodes_by_pos = {}
for pos in relevant_positions:
    v_nodes = edges_df[
        (edges_df['src'].str.contains(f'attn.v:{pos}:')) &
        (edges_df['is_cross_seq'] == True)
    ]['src'].unique()
    v_nodes_by_pos[pos] = set(v_nodes)

# Find shared and unique component indices
all_cidxs = {}
for pos, nodes in v_nodes_by_pos.items():
    for n in nodes:
        cidx = n.split(':')[2]
        all_cidxs.setdefault(cidx, set()).add(pos)

# Build asymmetry table: for each V component, sum(val) of all outgoing cross-seq edges per position
rows = []
for cidx, positions in sorted(all_cidxs.items(), key=lambda x: x[0]):
    row = {'cidx': cidx}
    for pos in relevant_positions:
        src_pattern = f'attn.v:{pos}:{cidx}'
        v_out = edges_df[
            (edges_df['src'].str.endswith(src_pattern)) &
            (edges_df['is_cross_seq'] == True)
        ]
        row[f'pos{pos}_sum'] = v_out['val'].sum() if len(v_out) > 0 else None
    rows.append(row)

asym_df = pd.DataFrame(rows)
# Compute diff between the two positions
pos_cols = [c for c in asym_df.columns if c.startswith('pos')]
if len(pos_cols) == 2:
    asym_df['diff'] = asym_df[pos_cols[0]].fillna(0) - asym_df[pos_cols[1]].fillna(0)
    asym_df = asym_df.sort_values('diff', ascending=False)
print(asym_df.to_string(index=False))
```

**What to look for:**
- **Sign flips**: Same V component produces opposite-sign contributions at two positions → token-identity detector
- **Position-unique components**: Fire at only one position → inherently asymmetric
- **Large |diff|**: Components contributing most to the prediction asymmetry

### Step 7: K → O analysis (attention routing)

Find K components feeding into the dominant O node. This reveals whether attention routing itself favors one source position.

```python
for o_node in dominant_o_nodes[:1]:  # focus on the most dominant
    k_edges = edges_df[
        (edges_df['tgt'] == o_node) &
        (edges_df['is_cross_seq'] == True) &
        (edges_df['src'].str.contains('attn.k'))
    ].copy()
    k_edges['src_pos'] = k_edges['src'].str.split(':').str[1].astype(int)
    k_edges['src_token'] = k_edges['src_pos'].map(lambda p: tokens[p] if p < len(tokens) else '?')

    print(f"\n=== K → {o_node} ===")
    pos_k = k_edges.groupby(['src_pos', 'src_token']).agg(sum_val=('val', 'sum'), n=('val', 'count'))
    print(pos_k.to_string())
```

Typically, **attention routing is roughly symmetric** — both variable positions get attended to equally. The asymmetry comes from the V side. But verify this.

### Step 8: Fetch autointerp for key components

For each important component identified above, fetch its LLM-generated interpretation.

```python
import requests

def get_interp(module: str, cidx: int) -> dict:
    """Fetch autointerp label for a component.

    Module names: graph nodes use short names (e.g. '2.attn.v'), but the API
    uses full module names. Conversion rules:
      - Attention sublayers: append '_proj' → '2.attn.v_proj'
      - MLP sublayers: append '_proj' → '0.mlp.down_proj'
    """
    resp = requests.get(f"http://localhost:{PORT}/api/correlations/interpretations/{module}/{cidx}")
    if resp.status_code == 200:
        data = resp.json()
        return {'label': data.get('label', '?'), 'reasoning': data.get('reasoning', ''), 'freq': data.get('activation_frequency')}
    return {'label': f'(no interp, status {resp.status_code})', 'reasoning': '', 'freq': None}

def node_to_module(node_key: str) -> tuple[str, int]:
    """Convert a graph node key like '2.attn.v:3:627' to (module, cidx).
    Returns ('2.attn.v_proj', 627)."""
    parts = node_key.split(':')
    layer = parts[0]
    cidx = int(parts[2])
    # Append _proj for attn and mlp sublayers
    if any(sub in layer for sub in ['attn.q', 'attn.k', 'attn.v', 'attn.o', 'mlp.up', 'mlp.down', 'mlp.gate']):
        layer = layer + '_proj'
    return layer, cidx
```

Call `get_interp` for the top V, K, and MLP components. Use the labels to annotate findings.

### Step 9: Upstream tracing

For the most important V components (especially those with large asymmetry or sign flips), trace what feeds them.

```python
for v_node in important_v_nodes:
    upstream = edges_df[edges_df['tgt'] == v_node].sort_values('abs_val', ascending=False)
    print(f"\n=== Upstream of {v_node} ===")
    print(upstream.head(10)[['src', 'val']].to_string(index=False))
```

Look for:
- MLP down components that differ between positions
- The **earliest layer** where asymmetry appears (e.g. `0.mlp.up` components active at only one position)
- Fetch autointerp for these upstream components too

### Step 10: Secondary pathways

Investigate other large contributors from step 3 that aren't the dominant pathway:
- Final-layer attention O nodes with negative attribution (suppression)
- Their V component inputs and what positions/tokens they attend to
- Whether they mainly drive MLP up (indirect path) or directly affect output

### Step 11: Competing tokens (optional)

Check if there's an output node for a "wrong" token. For example, if the target is ` y`, check for ` x`:

```python
# Find all output nodes in the graph
output_nodes = [n for n in set(edges_df['tgt']) | set(edges_df['src']) if n.startswith('output:')]
print("Output nodes in graph:", output_nodes)
```

If a competing output node exists, compare its attribution breakdown to the target to understand what pushes toward the correct vs incorrect prediction.

### Step 12: Write findings summary

Produce a markdown summary following this structure:

1. **Header**: Graph ID, prompt, target token, L0, edge count
2. **Output node attribution breakdown**: Layer summary table + top individual edges table
3. **Induction-like components analysis**:
   - V → O breakdown per dominant O node
   - V-side asymmetry table with diff column
   - Key asymmetry sources with autointerp labels
   - Upstream trace for the most important V components
   - K → O analysis
4. **Secondary pathways**: Other notable contributors (e.g. suppressive attention heads)
5. **Interpretation**: High-level summary of *why* the model predicts this token — what the dominant mechanism is, where asymmetry originates, and what the key components do

Save the findings to a file:

```python
findings_path = f"viz/graph_{graph_id}_findings.md"
with open(findings_path, 'w') as f:
    f.write(findings_text)
print(f"Findings saved to {findings_path}")
```

## Reference

### Node key format
- Regular nodes: `"layer:seq:cIdx"` (e.g. `2.attn.v:3:627`)
- Output nodes: `"output:seq:tokenId"` (e.g. `output:5:340`)

### Layer names
Graph nodes use short layer names: `embed`, `0.mlp.up`, `0.mlp.down`, `0.mlp.gate`, `0.attn.q`, `0.attn.k`, `0.attn.v`, `0.attn.o`, `1.mlp.up`, ..., `output`

### API module names
The autointerp API uses full module names with `_proj` suffix:
- Attention: `0.attn.q_proj`, `0.attn.k_proj`, `0.attn.v_proj`, `0.attn.o_proj`
- MLP: `0.mlp.up_proj`, `0.mlp.down_proj`, `0.mlp.gate_proj`

### Cross-sequence edges
Edges with `is_cross_seq=true` are attention edges (V→O and K→O). These connect nodes at different sequence positions and are the mechanism by which information flows across tokens.

### Useful pandas patterns

```python
# Parse node keys into components
edges_df[['src_layer', 'src_pos', 'src_cidx']] = edges_df['src'].str.rsplit(':', n=2, expand=True)

# Filter edges by target
edges_df[edges_df['tgt'] == target_node]

# Filter cross-sequence V edges
edges_df[(edges_df['is_cross_seq'] == True) & (edges_df['src'].str.contains('attn.v'))]

# Group by source layer with aggregations
out_edges.groupby('src_layer').agg(sum_val=('val', 'sum'), sum_abs=('abs_val', 'sum'), n=('val', 'count'))
```

### Interpreting attribution signs

Edge `val` is an attribution value, not a direct measure of positive/negative contribution to the output. The actual effect on the target logit depends on the **product of the component's activation and the attribution**:

- **Positive activation + positive attribution** → increases target logit
- **Negative activation + negative attribution** → also increases target logit (negative × negative = positive)
- **Positive activation + negative attribution** → decreases target logit
- **Negative activation + positive attribution** → also decreases target logit

Component activations are available in `graph_data['nodeSubcompActs']`. When interpreting whether an edge is "constructive" or "suppressive", check the source node's activation sign — a negative `val` from a negatively-activated component is actually a **positive** contribution.

### The fundamental question

The analysis answers: **"Why does the model predict token A instead of token B?"**

The answer usually comes from the **V side** (what values are written into the residual stream), not the **K side** (attention routing). Both candidate source positions typically get attended to roughly equally; the asymmetry is in the *values* written.

**Sign flips** are the clearest signal: when the same V component at two positions produces opposite-sign contributions, that component is a token-identity detector. It constructively contributes for the correct token and destructively for the wrong one.
