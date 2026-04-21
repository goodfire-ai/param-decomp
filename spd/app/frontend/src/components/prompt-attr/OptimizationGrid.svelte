<script lang="ts">
    import type { CISnapshot } from "../../lib/promptAttributionsTypes";

    type Props = {
        snapshot: CISnapshot;
    };

    let { snapshot }: Props = $props();

    let canvas: HTMLCanvasElement | undefined = $state();

    const LABEL_WIDTH = 64;
    const ROW_HEIGHT = 8;
    const ROW_GAP = 1;
    const GRID_WIDTH = 500;

    function abbreviateLayer(layer: string): string {
        const m = layer.match(/(\d+)\.\w+\.(\w+)/);
        if (!m) return layer;
        const shortNames: Record<string, string> = {
            q_proj: "q",
            k_proj: "k",
            v_proj: "v",
            o_proj: "o",
            up_proj: "up",
            gate_proj: "gate",
            down_proj: "down",
            c_fc: "up",
            c_proj: "down",
        };
        return `${m[1]}.${shortNames[m[2]] ?? m[2]}`;
    }

    const nLayers = $derived(snapshot.layers.length);
    const totalHeight = $derived(nLayers * (ROW_HEIGHT + ROW_GAP));
    const totalWidth = $derived(LABEL_WIDTH + GRID_WIDTH);

    // For each layer: total initial alive across all positions
    const rowTotals = $derived(snapshot.initial_alive.map((row) => row.reduce((a, b) => a + b, 0)));

    $effect(() => {
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = totalWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = `${totalWidth}px`;
        canvas.style.height = `${totalHeight}px`;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, totalWidth, totalHeight);

        // Draw layer labels
        ctx.font = "10px 'SF Mono', monospace";
        ctx.fillStyle = "#b4b4b4";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        for (let row = 0; row < nLayers; row++) {
            const y = row * (ROW_HEIGHT + ROW_GAP);
            ctx.fillText(abbreviateLayer(snapshot.layers[row]), LABEL_WIDTH - 4, y + ROW_HEIGHT / 2);

            const totalAlive = rowTotals[row];
            if (totalAlive === 0) continue;

            // Each cell = one initially-alive component, laid out left to right
            // grouped by position. Cell width = GRID_WIDTH / totalAlive.
            const cellWidth = GRID_WIDTH / totalAlive;
            let x = LABEL_WIDTH;

            for (let pos = 0; pos < snapshot.seq_len; pos++) {
                const initialAtPos = snapshot.initial_alive[row][pos];
                if (initialAtPos === 0) continue;

                const currentAtPos = snapshot.current_alive[row][pos];
                const fraction = currentAtPos / initialAtPos;

                // Draw the block for this position's alive components
                const blockWidth = cellWidth * initialAtPos;

                ctx.fillStyle = `rgba(124, 77, 51, ${Math.max(0.04, fraction)})`;
                ctx.fillRect(x, y, blockWidth - 0.5, ROW_HEIGHT);

                x += blockWidth;
            }
        }
    });

    const initialL0 = $derived(snapshot.initial_alive.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0));
    const fractionRemaining = $derived(initialL0 > 0 ? snapshot.l0_total / initialL0 : 0);
</script>

<div class="optimization-grid">
    <div class="grid-header">
        <span class="step-label">
            Step {snapshot.step}/{snapshot.total_steps}
        </span>
        <span class="l0-label">
            L0: {Math.round(snapshot.l0_total)} / {initialL0}
            ({(fractionRemaining * 100).toFixed(0)}%)
        </span>
        {#if snapshot.loss > 0}
            <span class="loss-label">loss: {snapshot.loss.toFixed(4)}</span>
        {/if}
    </div>
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .optimization-grid {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-2);
    }

    .grid-header {
        display: flex;
        gap: var(--space-4);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .l0-label {
        color: var(--accent-primary);
        font-weight: 600;
    }

    canvas {
        image-rendering: pixelated;
    }
</style>
