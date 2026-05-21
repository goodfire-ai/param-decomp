/**
 * Shared zoom/pan state and handlers for SVG graph visualizations.
 * - Shift + scroll to zoom
 * - Shift + drag (or middle-click drag) to pan
 */

const MIN_SCALE = 0.25;
const MAX_SCALE = 4;
const ZOOM_SENSITIVITY = 0.002;
const LINE_HEIGHT = 16;

export function useZoomPan(getContainer: () => HTMLElement | null) {
    let scale = $state(1);
    let translateX = $state(0);
    let translateY = $state(0);
    let isPanning = $state(false);
    let panStart: { x: number; y: number; tx: number; ty: number } | null = null;

    function zoom(centerX: number, centerY: number, factor: number) {
        const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
        if (newScale === scale) return;
        const ratio = newScale / scale;
        translateX = centerX - (centerX - translateX) * ratio;
        translateY = centerY - (centerY - translateY) * ratio;
        scale = newScale;
    }

    // Attach non-passive wheel listener for Shift+scroll zoom
    $effect(() => {
        const container = getContainer();
        if (!container) return;

        const handleWheel = (event: WheelEvent) => {
            if (!event.shiftKey) return;
            event.preventDefault();

            // Shift+wheel on some platforms converts deltaY to deltaX
            let delta = event.deltaY || event.deltaX;
            if (!delta) return;

            // Normalize to pixels
            if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) delta *= LINE_HEIGHT;
            else if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) delta *= container.clientHeight;

            const rect = container.getBoundingClientRect();
            zoom(
                event.clientX - rect.left + container.scrollLeft,
                event.clientY - rect.top + container.scrollTop,
                1 - delta * ZOOM_SENSITIVITY,
            );
        };

        container.addEventListener("wheel", handleWheel, { passive: false });
        return () => container.removeEventListener("wheel", handleWheel);
    });

    function startPan(event: MouseEvent) {
        event.preventDefault();
        isPanning = true;
        panStart = { x: event.clientX, y: event.clientY, tx: translateX, ty: translateY };
    }

    function updatePan(event: MouseEvent) {
        if (!isPanning || !panStart) return;
        translateX = panStart.tx + (event.clientX - panStart.x);
        translateY = panStart.ty + (event.clientY - panStart.y);
    }

    function endPan() {
        isPanning = false;
        panStart = null;
    }

    function zoomIn() {
        const container = getContainer();
        if (!container) return;
        zoom(container.clientWidth / 2 + container.scrollLeft, container.clientHeight / 2 + container.scrollTop, 1.25);
    }

    function zoomOut() {
        const container = getContainer();
        if (!container) return;
        zoom(container.clientWidth / 2 + container.scrollLeft, container.clientHeight / 2 + container.scrollTop, 0.8);
    }

    function reset() {
        scale = 1;
        translateX = 0;
        translateY = 0;
    }

    return {
        get scale() {
            return scale;
        },
        get translateX() {
            return translateX;
        },
        get translateY() {
            return translateY;
        },
        get isPanning() {
            return isPanning;
        },
        startPan,
        updatePan,
        endPan,
        zoomIn,
        zoomOut,
        reset,
    };
}
