<script lang="ts">
    import type { SubcomponentMetadata } from "../../lib/promptAttributionsTypes";

    interface Props {
        metadata: SubcomponentMetadata[];
        currentSubcomponentIdx: number | null;
        onSelect: (subcomponentIdx: number) => void;
    }

    let { metadata, currentSubcomponentIdx, onSelect }: Props = $props();

    const PLOT_HEIGHT = 200;
    const PLOT_PADDING = { top: 10, right: 15, bottom: 30, left: 55 };
    const LOG_Y_MIN = -6;

    let plotContainer = $state<HTMLDivElement | undefined>(undefined);
    let plotWidth = $state(600);
    let plotLogY = $state(true);

    $effect(() => {
        if (!plotContainer) return;

        const observer = new ResizeObserver((entries) => {
            plotWidth = entries[0].contentRect.width;
        });
        observer.observe(plotContainer);
        return () => observer.disconnect();
    });

    const plotData = $derived.by(() => {
        if (metadata.length === 0) return null;

        const innerWidth = plotWidth - PLOT_PADDING.left - PLOT_PADDING.right;
        const innerHeight = PLOT_HEIGHT - PLOT_PADDING.top - PLOT_PADDING.bottom;
        const n = metadata.length;
        const xScale = n > 1 ? innerWidth / (n - 1) : 0;

        if (plotLogY) {
            const logValues = metadata.map((m) => Math.log10(Math.max(m.mean_ci, 1e-20)));
            const yMin = Math.max(LOG_Y_MIN, Math.min(...logValues));
            const yMax = 0;
            const yRange = yMax - yMin || 1;

            const points = logValues.map((logVal, i) => ({
                x: PLOT_PADDING.left + i * xScale,
                y: PLOT_PADDING.top + (1 - (Math.max(logVal, yMin) - yMin) / yRange) * innerHeight,
                rank: i,
            }));

            const yTicks: { value: number; y: number; label: string }[] = [];
            for (let v = 0; v >= yMin; v -= 2) {
                yTicks.push({
                    value: v,
                    y: PLOT_PADDING.top + (1 - (v - yMin) / yRange) * innerHeight,
                    label: `1e${v}`,
                });
            }

            return { points, yTicks, innerWidth, innerHeight, n };
        }

        const points = metadata.map((m, i) => ({
            x: PLOT_PADDING.left + i * xScale,
            y: PLOT_PADDING.top + (1 - m.mean_ci) * innerHeight,
            rank: i,
        }));
        const yTicks: { value: number; y: number; label: string }[] = [
            { value: 1, y: PLOT_PADDING.top, label: "1" },
            { value: 0, y: PLOT_PADDING.top + innerHeight, label: "0" },
        ];

        return { points, yTicks, innerWidth, innerHeight, n };
    });

    const currentPointIndex = $derived.by(() => {
        if (currentSubcomponentIdx === null) return null;
        return metadata.findIndex((m) => m.subcomponent_idx === currentSubcomponentIdx);
    });

    function handlePlotClick(rank: number) {
        const subcomponent = metadata[rank];
        if (!subcomponent) return;
        onSelect(subcomponent.subcomponent_idx);
    }
</script>

<div class="ci-plot" bind:this={plotContainer}>
    <label class="plot-toggle">
        <input type="checkbox" bind:checked={plotLogY} />
        Log Y
    </label>
    {#if plotData}
        <svg width={plotWidth} height={PLOT_HEIGHT}>
            {#each plotData.yTicks as tick (tick.value)}
                <line
                    x1={PLOT_PADDING.left}
                    y1={tick.y}
                    x2={PLOT_PADDING.left + plotData.innerWidth}
                    y2={tick.y}
                    stroke="var(--border-subtle)"
                    stroke-width="1"
                />
                <text
                    x={PLOT_PADDING.left - 8}
                    y={tick.y}
                    text-anchor="end"
                    dominant-baseline="middle"
                    class="plot-label"
                >
                    {tick.label}
                </text>
            {/each}

            {#if plotData.points.length > 1}
                <polyline
                    points={plotData.points.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none"
                    stroke="var(--accent-primary)"
                    stroke-width="1.5"
                />
            {/if}

            {#each plotData.points as point (point.rank)}
                <rect
                    x={point.x - Math.max(plotData.innerWidth / plotData.n / 2, 2)}
                    y={PLOT_PADDING.top}
                    width={Math.max(plotData.innerWidth / plotData.n, 4)}
                    height={plotData.innerHeight}
                    fill="transparent"
                    class="plot-hitarea"
                    onclick={() => handlePlotClick(point.rank)}
                />
            {/each}

            {#if currentPointIndex !== null && plotData.points[currentPointIndex]}
                {@const cp = plotData.points[currentPointIndex]}
                <line
                    x1={cp.x}
                    y1={PLOT_PADDING.top}
                    x2={cp.x}
                    y2={PLOT_PADDING.top + plotData.innerHeight}
                    stroke="var(--accent-primary-dim)"
                    stroke-width="1"
                    stroke-dasharray="3 2"
                />
                <circle cx={cp.x} cy={cp.y} r="4" fill="var(--accent-primary)" />
            {/if}

            <text
                x={PLOT_PADDING.left + plotData.innerWidth / 2}
                y={PLOT_HEIGHT - 4}
                text-anchor="middle"
                class="plot-label"
            >
                Component rank ({plotData.n} total)
            </text>
        </svg>
    {/if}
</div>

<style>
    .ci-plot {
        position: relative;
        width: 100%;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
    }

    .plot-toggle {
        position: absolute;
        top: var(--space-1);
        right: var(--space-2);
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        cursor: pointer;
        z-index: 1;
    }

    .plot-toggle input {
        cursor: pointer;
    }

    .ci-plot svg {
        display: block;
    }

    .plot-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        fill: var(--text-muted);
    }

    .plot-hitarea {
        cursor: pointer;
    }
</style>
