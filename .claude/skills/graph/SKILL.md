---
name: graph
description: Load an attribution graph from the SPD app into a Scribe notebook for analysis
user-invocable: true
---

# Load Attribution Graph

Load an attribution graph from a running SPD app instance into a Scribe notebook session for interactive analysis.

## Usage

`/graph <port> [graph_id]`

- `port` (required): The port the SPD app backend is running on (e.g. `8000`)
- `graph_id` (optional): The specific graph ID to load. If omitted, list all available prompts and their graphs so the user can choose.

## Steps

1. **Start a Scribe session** using `mcp__scribe__start_new_session` with experiment name `AttributionGraphAnalysis`.

2. **If no graph_id provided**, discover what's available:
   - `GET http://localhost:{port}/api/prompts` → list of `{id, tokens, preview}` for each prompt
   - For each prompt, `GET http://localhost:{port}/api/graphs/{prompt_id}?normalize=none&ci_threshold=0.0` → list of graphs with their IDs
   - Show the user a summary table of available graphs (graph ID, prompt preview, graph type, L0) and ask which one to load.

3. **Fetch the graph** using the agents endpoint which takes a graph ID directly:
   ```
   GET http://localhost:{port}/api/agents/graphs/{graph_id}?normalize=none&ci_threshold=0.0
   ```
   Required query parameters:
   - `normalize`: one of `none`, `target`, `layer`. Default to `none` (raw attribution values).
   - `ci_threshold`: float >= 0. Default to `0.0` (include all components).

4. **Parse the response** into convenient DataFrames in the notebook. The response schema is:
   ```
   {
     id: int,
     graphType: "standard" | "optimized",
     tokens: list[str],              # token strings for each seq position
     edges: list[{src, tgt, val, is_cross_seq}],  # "layer:seq:cIdx" format
     outputProbs: dict[str, {prob, token}],        # "seq:cIdx" → prediction
     nodeCiVals: dict[str, float],                 # "layer:seq:cIdx" → CI value
     nodeSubcompActs: dict[str, float],            # "layer:seq:cIdx" → activation
     maxAbsAttr: float,
     maxAbsSubcompAct: float,
     l0_total: int,
     optimization?: {imp_min_coeff, steps, pnorm, beta, mask_type, loss, metrics, pgd}
   }
   ```

5. **Set up the notebook** with these variables available for subsequent analysis:
   - `graph_data`: the raw JSON response dict
   - `edges_df`: DataFrame of edges with columns `src, tgt, val, is_cross_seq, abs_val`
   - `tokens`: list of token strings
   - `ci_vals`: dict of node CI values
   - `PORT`: the port number for follow-up API calls

6. **Print a summary** showing:
   - Graph ID, type, prompt tokens
   - Optimization info (if optimized): loss config, label token, key metrics
   - Total edges, active components (L0), max |attribution|
   - Active components per layer
   - Active components per sequence position
   - Top 10 edges by |val|

## Important notes

- The app backend must already be running at the specified port.
- Node keys follow the format `"layer:seq:cIdx"` where layer names are like `embed`, `0.mlp.up`, `2.attn.o`, `output`.
- Output nodes use `"output:seq:token_id"` where `token_id` is the vocab index.
- Cross-sequence edges (`is_cross_seq=true`) represent attention patterns (k/v → o_proj).
- The `normalize` and `ci_threshold` params can be changed in follow-up API calls if the user wants different views. Store them as variables.