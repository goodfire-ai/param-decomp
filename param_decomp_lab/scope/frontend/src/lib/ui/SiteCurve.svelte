<script lang="ts">
    import { goto } from "$app/navigation";
    import type { SiteCurve } from "$lib/types";

    interface Props {
        curve: SiteCurve;
        currentRank: number | null;
        run: string;
        site: string;
    }
    let { curve, currentRank, run, site }: Props = $props();

    const HEIGHT = 150;
    const PAD = { top: 12, right: 14, bottom: 26, left: 52 };
    const LOG_FLOOR = -6;

    let container = $state<HTMLDivElement | undefined>(undefined);
    let width = $state(700);
    let logY = $state(true);

    $effect(() => {
        if (!container) return;
        const observer = new ResizeObserver((entries) => {
            width = entries[0].contentRect.width;
        });
        observer.observe(container);
        return () => observer.disconnect();
    });

    const innerW = $derived(width - PAD.left - PAD.right);
    const innerH = $derived(HEIGHT - PAD.top - PAD.bottom);

    function yOf(meanCi: number): number {
        if (logY) {
            const v = Math.max(Math.log10(Math.max(meanCi, 1e-20)), LOG_FLOOR);
            return PAD.top + (1 - (v - LOG_FLOOR) / -LOG_FLOOR) * innerH;
        }
        return PAD.top + (1 - meanCi) * innerH;
    }
    const xOf = (rank: number) => PAD.left + (rank / Math.max(curve.total - 1, 1)) * innerW;

    const path = $derived(curve.points.map((p) => `${xOf(p.rank)},${yOf(p.mean_ci)}`).join(" "));
    const yTicks = $derived(
        logY
            ? [0, -2, -4, -6].map((v) => ({
                  y: PAD.top + (1 - (v - LOG_FLOOR) / -LOG_FLOOR) * innerH,
                  label: v === 0 ? "1" : `1e${v}`,
              }))
            : [
                  { y: PAD.top, label: "1" },
                  { y: PAD.top + innerH, label: "0" },
              ],
    );

    const currentPoint = $derived.by(() => {
        if (currentRank === null) return null;
        let best = curve.points[0];
        for (const p of curve.points) {
            if (Math.abs(p.rank - currentRank) < Math.abs(best.rank - currentRank)) best = p;
        }
        return best;
    });

    function onClick(event: MouseEvent) {
        const box = (event.currentTarget as SVGElement).getBoundingClientRect();
        const rank = ((event.clientX - box.left - PAD.left) / innerW) * (curve.total - 1);
        let best = curve.points[0];
        for (const p of curve.points) {
            if (Math.abs(p.rank - rank) < Math.abs(best.rank - rank)) best = p;
        }
        goto(`/r/${run}/s/${site}/c/${best.idx}`);
    }
</script>

<div class="curve" bind:this={container}>
    <button class="log-toggle num" class:active={logY} onclick={() => (logY = !logY)}>
        log y
    </button>
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
    <svg {width} height={HEIGHT} onclick={onClick} role="button" tabindex="-1">
        {#each yTicks as tick (tick.label)}
            <line x1={PAD.left} y1={tick.y} x2={PAD.left + innerW} y2={tick.y} class="gridline" />
            <text x={PAD.left - 7} y={tick.y} text-anchor="end" dominant-baseline="middle" class="tick">
                {tick.label}
            </text>
        {/each}
        <polyline points={path} class="line" />
        {#if currentPoint !== null}
            <line
                x1={xOf(currentPoint.rank)}
                y1={PAD.top}
                x2={xOf(currentPoint.rank)}
                y2={PAD.top + innerH}
                class="marker-line"
            />
            <circle
                cx={xOf(currentPoint.rank)}
                cy={yOf(currentPoint.mean_ci)}
                r="3.5"
                class="marker"
            />
        {/if}
        <text x={PAD.left + innerW / 2} y={HEIGHT - 4} text-anchor="middle" class="tick">
            mean causal importance by component rank ({curve.total.toLocaleString()} components — click to jump)
        </text>
    </svg>
</div>

<style>
    .curve {
        position: relative;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 5px;
        margin-bottom: 24px;
    }
    .log-toggle {
        position: absolute;
        top: 8px;
        right: 10px;
        font-size: 10px;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 2px 8px;
        background: var(--panel);
        color: var(--dim);
        cursor: pointer;
    }
    .log-toggle:hover {
        border-color: var(--line-2);
    }
    .log-toggle.active {
        color: var(--fg);
        border-color: var(--line-2);
    }
    svg {
        display: block;
        cursor: crosshair;
    }
    .gridline {
        stroke: var(--line);
        stroke-width: 1;
    }
    .tick {
        font-family: var(--mono);
        font-size: 10px;
        fill: var(--faint);
    }
    .line {
        fill: none;
        stroke: var(--accent);
        stroke-width: 1.4;
    }
    .marker-line {
        stroke: var(--accent);
        stroke-width: 1;
        stroke-dasharray: 3 3;
        opacity: 0.6;
    }
    .marker {
        fill: var(--accent);
    }
</style>
