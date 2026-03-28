<script lang="ts">
    import type { FullMatrixResponse, FullMatrixMetric, ClusterBoundary } from "../lib/api/clusters";
    import { fetchFullMatrix } from "../lib/api/clusters";
    import type { ClusterMappingData } from "../lib/useRun.svelte";

    type Props = {
        clusterMappingData: ClusterMappingData;
        onSelectCluster: (clusterId: number) => void;
    };

    let { clusterMappingData, onSelectCluster }: Props = $props();

    let selectedMetric = $state<FullMatrixMetric>("jaccard");

    type LoadedData = { response: FullMatrixResponse; floats: Float32Array };
    type DataState =
        | { status: "idle" }
        | { status: "loading" }
        | { status: "loaded"; data: LoadedData }
        | { status: "error"; error: string };

    let overviewState = $state<DataState>({ status: "idle" });
    let detailState = $state<DataState>({ status: "idle" });

    let overviewCanvas = $state<HTMLCanvasElement | null>(null);
    let detailCanvas = $state<HTMLCanvasElement | null>(null);

    // Selection rectangle on overview (in overview pixel coords)
    let selRow = $state(0);
    let selCol = $state(0);
    let isDragging = $state(false);

    const OVERVIEW_SIZE = 700;
    const DETAIL_SIZE = 500;
    const DETAIL_REGION = 200;

    let hoverInfo = $state<{
        x: number;
        y: number;
        rowKey: string;
        colKey: string;
        value: number | null;
        cluster: ClusterBoundary | null;
    } | null>(null);

    function decodeMatrix(response: FullMatrixResponse): Float32Array {
        const binaryStr = atob(response.matrix_b64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
        }
        return new Float32Array(bytes.buffer);
    }

    // Fetch overview (small, binned)
    $effect(() => {
        const metric = selectedMetric;
        const mapping = clusterMappingData;
        overviewState = { status: "loading" };
        detailState = { status: "idle" };
        selRow = 0;
        selCol = 0;
        fetchFullMatrix(metric, mapping, 500)
            .then((response) => {
                overviewState = { status: "loaded", data: { response, floats: decodeMatrix(response) } };
            })
            .catch((e) => (overviewState = { status: "error", error: String(e) }));
    });

    // Debounced detail fetch
    let detailTimer: ReturnType<typeof setTimeout> | null = null;

    $effect(() => {
        if (overviewState.status !== "loaded") return;
        // Read reactive deps
        const fullSize = overviewState.data.response.full_size;
        const row = Math.round((selRow * fullSize) / overviewState.data.response.height);
        const col = Math.round((selCol * fullSize) / overviewState.data.response.width);
        const metric = selectedMetric;
        const mapping = clusterMappingData;

        if (detailTimer !== null) clearTimeout(detailTimer);
        detailTimer = setTimeout(() => {
            detailState = { status: "loading" };
            fetchFullMatrix(metric, mapping, DETAIL_REGION, { row, col, size: DETAIL_REGION })
                .then((response) => {
                    detailState = { status: "loaded", data: { response, floats: decodeMatrix(response) } };
                })
                .catch((e) => (detailState = { status: "error", error: String(e) }));
        }, 150);
    });

    // Render overview image once when data loads
    let overviewImageData = $state<ImageData | null>(null);

    $effect(() => {
        if (overviewState.status !== "loaded") return;
        const { response, floats } = overviewState.data;
        const imgData = new ImageData(response.width, response.height);
        const pixels = imgData.data;
        for (let i = 0; i < floats.length; i++) {
            const [r, g, b] = metricToColor(floats[i], selectedMetric);
            const off = i * 4;
            pixels[off] = r;
            pixels[off + 1] = g;
            pixels[off + 2] = b;
            pixels[off + 3] = 255;
        }
        overviewImageData = imgData;
    });

    // Redraw overview canvas (image + boundaries + selection rect) when selection moves
    $effect(() => {
        if (!overviewCanvas || !overviewImageData || overviewState.status !== "loaded") return;
        const ov = overviewState.data.response;
        overviewCanvas.width = ov.width;
        overviewCanvas.height = ov.height;
        const ctx = overviewCanvas.getContext("2d")!;
        ctx.putImageData(overviewImageData, 0, 0);

        // Cluster boundaries (overview is always the full symmetric matrix, so row=col)
        ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
        ctx.lineWidth = 1;
        for (const b of ov.row_boundaries) {
            ctx.strokeRect(b.start, b.start, b.end - b.start, b.end - b.start);
        }

        // Selection rectangle
        const _row = selRow;
        const _col = selCol;
        const selW = Math.round((DETAIL_REGION * ov.width) / ov.full_size);
        const selH = Math.round((DETAIL_REGION * ov.height) / ov.full_size);
        ctx.strokeStyle = "rgba(255, 200, 0, 0.9)";
        ctx.lineWidth = 2;
        ctx.strokeRect(_col, _row, selW, selH);
    });

    // Render detail
    $effect(() => {
        if (detailState.status !== "loaded" || !detailCanvas) return;
        const { response, floats } = detailState.data;
        detailCanvas.width = response.width;
        detailCanvas.height = response.height;
        renderCanvas(
            detailCanvas,
            floats,
            response.width,
            response.height,
            response.row_boundaries,
            response.col_boundaries,
            selectedMetric,
        );
    });

    function metricToColor(val: number, metric: FullMatrixMetric): [number, number, number] {
        if (isNaN(val)) return [240, 240, 240];
        if (metric === "pmi") {
            if (val > 0) {
                const n = Math.min(val / 6, 1);
                return [
                    Math.round(59 * n + 255 * (1 - n)),
                    Math.round(130 * n + 255 * (1 - n)),
                    Math.round(246 * n + 255 * (1 - n)),
                ];
            } else {
                const n = Math.min(Math.abs(val) / 6, 1);
                return [
                    Math.round(239 * n + 255 * (1 - n)),
                    Math.round(68 * n + 255 * (1 - n)),
                    Math.round(68 * n + 255 * (1 - n)),
                ];
            }
        }
        const n = Math.min(Math.max(val, 0), 1);
        return [
            Math.round(59 * n + 255 * (1 - n)),
            Math.round(130 * n + 255 * (1 - n)),
            Math.round(246 * n + 255 * (1 - n)),
        ];
    }

    function renderCanvas(
        cvs: HTMLCanvasElement,
        floats: Float32Array,
        width: number,
        height: number,
        rowBoundaries: ClusterBoundary[],
        colBoundaries: ClusterBoundary[],
        metric: FullMatrixMetric,
    ) {
        cvs.width = width;
        cvs.height = height;
        const ctx = cvs.getContext("2d")!;
        const imageData = ctx.createImageData(width, height);
        const pixels = imageData.data;
        for (let i = 0; i < floats.length; i++) {
            const [r, g, b] = metricToColor(floats[i], metric);
            const off = i * 4;
            pixels[off] = r;
            pixels[off + 1] = g;
            pixels[off + 2] = b;
            pixels[off + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);

        ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
        ctx.lineWidth = 1;
        // Horizontal lines at row boundaries
        for (const b of rowBoundaries) {
            ctx.beginPath();
            ctx.moveTo(0, b.start);
            ctx.lineTo(width, b.start);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, b.end);
            ctx.lineTo(width, b.end);
            ctx.stroke();
        }
        // Vertical lines at col boundaries
        for (const b of colBoundaries) {
            ctx.beginPath();
            ctx.moveTo(b.start, 0);
            ctx.lineTo(b.start, height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(b.end, 0);
            ctx.lineTo(b.end, height);
            ctx.stroke();
        }
    }

    function overviewToSel(e: MouseEvent) {
        if (!overviewCanvas || overviewState.status !== "loaded") return;
        const rect = overviewCanvas.getBoundingClientRect();
        const ov = overviewState.data.response;
        const scaleX = ov.width / rect.width;
        const scaleY = ov.height / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        const selW = Math.round((DETAIL_REGION * ov.width) / ov.full_size);
        const selH = Math.round((DETAIL_REGION * ov.height) / ov.full_size);
        selCol = Math.max(0, Math.min(Math.round(mx - selW / 2), ov.width - selW));
        selRow = Math.max(0, Math.min(Math.round(my - selH / 2), ov.height - selH));
    }

    function handleOverviewMouseDown(e: MouseEvent) {
        if (e.button !== 0) return;
        isDragging = true;
        overviewToSel(e);
    }

    function handleOverviewMouseMove(e: MouseEvent) {
        if (isDragging) overviewToSel(e);
    }

    function handleOverviewMouseUp() {
        isDragging = false;
    }

    function handleDetailMouseMove(e: MouseEvent) {
        if (detailState.status !== "loaded" || !detailCanvas) {
            hoverInfo = null;
            return;
        }
        const rect = detailCanvas.getBoundingClientRect();
        const d = detailState.data.response;
        const scaleX = d.width / rect.width;
        const scaleY = d.height / rect.height;
        const matX = Math.floor((e.clientX - rect.left) * scaleX);
        const matY = Math.floor((e.clientY - rect.top) * scaleY);
        if (matX < 0 || matX >= d.width || matY < 0 || matY >= d.height) {
            hoverInfo = null;
            return;
        }
        const idx = matY * d.width + matX;
        const val = detailState.data.floats[idx];
        const colKey = matX < d.component_keys.length ? d.component_keys[matX] : "";
        const rowKey = matY < d.component_keys.length ? d.component_keys[matY] : "";
        const rowCluster = d.row_boundaries.find((b) => matY >= b.start && matY < b.end) ?? null;
        const colCluster = d.col_boundaries.find((b) => matX >= b.start && matX < b.end) ?? null;
        const cluster = rowCluster && colCluster && rowCluster.cluster_id === colCluster.cluster_id ? rowCluster : null;
        hoverInfo = { x: e.clientX, y: e.clientY, rowKey, colKey, value: isNaN(val) ? null : val, cluster };
    }

    function handleDetailClick(e: MouseEvent) {
        if (detailState.status !== "loaded" || !detailCanvas) return;
        const rect = detailCanvas.getBoundingClientRect();
        const d = detailState.data.response;
        const scaleX = d.width / rect.width;
        const scaleY = d.height / rect.height;
        const matX = Math.floor((e.clientX - rect.left) * scaleX);
        const matY = Math.floor((e.clientY - rect.top) * scaleY);
        const rowCluster = d.row_boundaries.find((b) => matY >= b.start && matY < b.end);
        const colCluster = d.col_boundaries.find((b) => matX >= b.start && matX < b.end);
        const cluster = rowCluster && colCluster && rowCluster.cluster_id === colCluster.cluster_id ? rowCluster : null;
        if (cluster) onSelectCluster(cluster.cluster_id);
    }
</script>

<svelte:window onmouseup={handleOverviewMouseUp} />

<div class="full-matrix">
    <div class="matrix-header">
        <div class="matrix-controls">
            <span class="control-label">Metric:</span>
            <button
                class="metric-btn"
                class:active={selectedMetric === "jaccard"}
                onclick={() => (selectedMetric = "jaccard")}>Jaccard</button
            >
            <button
                class="metric-btn"
                class:active={selectedMetric === "precision"}
                onclick={() => (selectedMetric = "precision")}>Precision</button
            >
            <button class="metric-btn" class:active={selectedMetric === "pmi"} onclick={() => (selectedMetric = "pmi")}
                >PMI</button
            >
        </div>
        {#if overviewState.status === "loaded"}
            <span class="info-text">
                {overviewState.data.response.full_size}×{overviewState.data.response.full_size} components · {overviewState.data.response.n_tokens.toLocaleString()}
                tokens · {overviewState.data.response.row_boundaries.length} clusters
            </span>
        {/if}
    </div>

    {#if overviewState.status === "loading"}
        <div class="status">Loading overview...</div>
    {:else if overviewState.status === "error"}
        <div class="status error">{overviewState.error}</div>
    {:else if overviewState.status === "loaded"}
        <div class="matrix-panels">
            <div class="panel overview-panel">
                <div class="panel-label">Overview (click to select region)</div>
                <canvas
                    bind:this={overviewCanvas}
                    style="width: {OVERVIEW_SIZE}px; height: {OVERVIEW_SIZE}px;"
                    class="matrix-canvas overview"
                    class:dragging={isDragging}
                    onmousedown={handleOverviewMouseDown}
                    onmousemove={handleOverviewMouseMove}
                ></canvas>
            </div>
            <div class="panel detail-panel">
                <div class="panel-label">
                    Detail ({DETAIL_REGION}×{DETAIL_REGION})
                    {#if detailState.status === "loading"}
                        <span class="loading-dot">loading...</span>
                    {/if}
                </div>
                {#if detailState.status === "loaded"}
                    <div class="detail-canvas-wrapper">
                        <canvas
                            bind:this={detailCanvas}
                            style="width: {DETAIL_SIZE}px; height: {DETAIL_SIZE}px;"
                            class="matrix-canvas detail"
                            onmousemove={handleDetailMouseMove}
                            onmouseleave={() => (hoverInfo = null)}
                            onclick={handleDetailClick}
                        ></canvas>
                        {#if hoverInfo}
                            <div class="tooltip" style="left: {hoverInfo.x + 12}px; top: {hoverInfo.y - 40}px;">
                                <div class="tooltip-row">
                                    <span class="tooltip-label">row</span><span class="tooltip-key"
                                        >{hoverInfo.rowKey}</span
                                    >
                                </div>
                                <div class="tooltip-row">
                                    <span class="tooltip-label">col</span><span class="tooltip-key"
                                        >{hoverInfo.colKey}</span
                                    >
                                </div>
                                <div class="tooltip-row">
                                    <span class="tooltip-label">{selectedMetric}</span>
                                    <span class="tooltip-val"
                                        >{hoverInfo.value !== null ? hoverInfo.value.toFixed(4) : "—"}</span
                                    >
                                    {#if hoverInfo.cluster}<span class="tooltip-cluster"
                                            >Cluster {hoverInfo.cluster.cluster_id}</span
                                        >{/if}
                                </div>
                            </div>
                        {/if}
                    </div>
                {:else if detailState.status === "error"}
                    <div class="status error">{detailState.error}</div>
                {/if}
            </div>
        </div>
        <div class="matrix-footer">Click a cluster block in the detail view to inspect</div>
    {/if}
</div>

<style>
    .full-matrix {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        height: 100%;
        min-height: 0;
    }

    .matrix-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-shrink: 0;
    }

    .matrix-controls {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .info-text {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .control-label {
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .metric-btn {
        padding: var(--space-1) var(--space-3);
        font: inherit;
        font-size: var(--text-sm);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        color: var(--text-secondary);
        transition:
            background var(--transition-normal),
            border-color var(--transition-normal);
    }

    .metric-btn:hover {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
    }

    .metric-btn.active {
        background: var(--bg-elevated);
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }

    .status {
        font-size: var(--text-sm);
        color: var(--text-muted);
        padding: var(--space-4);
    }

    .status.error {
        color: var(--status-error);
    }

    .matrix-panels {
        display: flex;
        gap: var(--space-4);
        flex: 1;
        min-height: 0;
        align-items: flex-start;
    }

    .panel {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .panel-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-weight: 500;
    }

    .loading-dot {
        color: var(--text-muted);
        font-style: italic;
    }

    .matrix-canvas {
        border: 1px solid var(--border-default);
        image-rendering: pixelated;
    }

    .matrix-canvas.overview {
        cursor: crosshair;
    }

    .matrix-canvas.overview.dragging {
        cursor: grabbing;
    }

    .matrix-canvas.detail {
        cursor: crosshair;
    }

    .detail-canvas-wrapper {
        position: relative;
    }

    .tooltip {
        position: fixed;
        z-index: 1000;
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        pointer-events: none;
        display: flex;
        flex-direction: column;
        gap: 2px;
        box-shadow: var(--shadow-md);
        white-space: nowrap;
    }

    .tooltip-row {
        display: flex;
        gap: var(--space-2);
        align-items: center;
    }

    .tooltip-label {
        color: var(--text-muted);
        min-width: 2em;
    }

    .tooltip-key {
        color: var(--text-secondary);
        max-width: 250px;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .tooltip-val {
        color: var(--text-primary);
        font-weight: 600;
    }

    .tooltip-cluster {
        color: var(--accent-primary);
    }

    .matrix-footer {
        font-size: var(--text-xs);
        color: var(--text-muted);
        flex-shrink: 0;
    }
</style>
