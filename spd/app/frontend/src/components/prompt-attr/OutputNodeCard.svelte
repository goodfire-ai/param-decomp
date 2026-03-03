<script lang="ts">
    import type { OutputProbability } from "../../lib/promptAttributionsTypes";
    import { getOutputHeaderColor } from "../../lib/colors";

    type Props = {
        cIdx: number;
        outputProbs: Record<string, OutputProbability>;
        seqIdx?: number; // When present: show single position. When absent: show all positions for this vocab id
    };

    let { cIdx, outputProbs, seqIdx }: Props = $props();

    function escapeHtml(text: string): string {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    // Single position mode
    const singlePosEntry = $derived.by(() => {
        if (seqIdx === undefined) return null;
        const entry = outputProbs[`${seqIdx}:${cIdx}`];
        if (!entry) throw new Error(`OutputNodeCard: no entry for ${seqIdx}:${cIdx}`);
        return entry;
    });

    // All positions mode - find all positions where this vocab id appears
    const allPositions = $derived.by(() => {
        if (seqIdx !== undefined) return null;
        const positions = Object.entries(outputProbs)
            .filter(([key]) => key.endsWith(`:${cIdx}`))
            .map(([key, entry]) => ({
                seqIdx: parseInt(key.split(":")[0]),
                prob: entry.prob,
                logit: entry.logit,
                target_prob: entry.target_prob,
                target_logit: entry.target_logit,
                token: entry.token,
            }))
            .sort((a, b) => b.prob - a.prob);
        if (positions.length === 0) throw new Error(`OutputNodeCard: no positions for cIdx ${cIdx}`);
        return positions;
    });
</script>

<div class="output-node-card">
    {#if singlePosEntry}
        <div class="output-header" style="background: {getOutputHeaderColor(singlePosEntry.prob)};">
            <div class="output-token">"{escapeHtml(singlePosEntry.token)}"</div>
            <div class="output-prob">
                CI-masked: {(singlePosEntry.prob * 100).toFixed(1)}% (logit: {singlePosEntry.logit.toFixed(2)})
            </div>
            <div class="output-prob">
                Target: {(singlePosEntry.target_prob * 100).toFixed(1)}% (logit: {singlePosEntry.target_logit.toFixed(
                    2,
                )})
            </div>
        </div>
        <p class="stats">
            <strong>Position:</strong>
            {seqIdx}
        </p>
    {:else if allPositions}
        <p><strong>"{allPositions[0].token}"</strong></p>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Pos</th>
                    <th>CI-masked</th>
                    <th>Logit</th>
                    <th>Target</th>
                    <th>Logit</th>
                </tr>
            </thead>
            <tbody>
                {#each allPositions as pos (pos.seqIdx)}
                    <tr>
                        <td>{pos.seqIdx}</td>
                        <td>{(pos.prob * 100).toFixed(2)}%</td>
                        <td>{pos.logit.toFixed(2)}</td>
                        <td>{(pos.target_prob * 100).toFixed(2)}%</td>
                        <td>{pos.target_logit.toFixed(2)}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
</div>

<style>
    .output-node-card {
        font-size: var(--text-base);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .output-header {
        padding: var(--space-2) var(--space-3);
        margin-bottom: var(--space-2);
        border-left: 2px solid var(--status-positive);
    }

    .output-token {
        font-size: 1.1em;
        font-weight: 600;
        font-family: var(--font-mono);
        color: var(--text-primary);
    }

    .output-prob {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    .stats {
        margin: var(--space-1) 0;
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    .stats strong {
        color: var(--text-muted);
        font-weight: 500;
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
    }

    .data-table {
        font-size: var(--text-sm);
        margin-top: var(--space-1);
        font-family: var(--font-mono);
        border-collapse: collapse;
    }

    .data-table th {
        padding: var(--space-1) var(--space-2);
        color: var(--text-muted);
        font-weight: 500;
        font-size: var(--text-xs);
        text-align: left;
        border-bottom: 1px solid var(--border-subtle);
    }

    .data-table th:not(:first-child) {
        text-align: right;
    }

    .data-table td {
        padding: var(--space-1) var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .data-table td:first-child {
        color: var(--text-primary);
    }

    .data-table td:not(:first-child) {
        color: var(--text-secondary);
        text-align: right;
    }
</style>
