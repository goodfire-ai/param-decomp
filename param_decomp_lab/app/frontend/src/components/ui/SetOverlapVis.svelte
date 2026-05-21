<script lang="ts">
    import { colors, rgbToCss } from "../../lib/colors";

    type Props = {
        /** Number of items in set A (subject/query component) */
        countA: number;
        /** Number of items in set B (object/other component) */
        countB: number;
        /** Number of items in intersection A ∩ B */
        countIntersection: number;
        /** Total count to scale against (e.g. total tokens in dataset) */
        totalCount: number;
        /** Scale bar relative to population (shows white "rest") or union only */
        relativeTo?: "population" | "union";
    };

    let { countA, countB, countIntersection, totalCount, relativeTo = "population" }: Props = $props();

    const countAOnly = $derived(Math.max(0, countA - countIntersection));
    const countBOnly = $derived(Math.max(0, countB - countIntersection));
    const countUnion = $derived(countAOnly + countIntersection + countBOnly);

    // Population-relative percentages (includes white "rest" background)
    const pctUnionPop = $derived((countUnion / totalCount) * 100);
    const pctAWithIntersectionPop = $derived(((countAOnly + countIntersection) / totalCount) * 100);
    const pctAOnlyPop = $derived((countAOnly / totalCount) * 100);

    // Union-relative percentages (scaled to 100% of union, no "rest")
    // Add 5 to each non-zero segment then normalize - boosts visibility of small sections
    const BOOST = 3;
    const unionSegments = $derived.by(() => {
        if (countUnion === 0) return { aOnly: 0, intersection: 0, bOnly: 0 };

        const baseAOverUnion = (countAOnly / countUnion) * 100;
        const baseIntersectionOverUnion = (countIntersection / countUnion) * 100;
        const baseBOverUnion = (countBOnly / countUnion) * 100;
        if (countAOnly + countIntersection + countBOnly !== countUnion) {
            throw new Error(
                "countAOnly + countIntersection + countBOnly !== countUnion" +
                    `: ${countAOnly} + ${countIntersection} + ${countBOnly} !== ${countUnion}`,
            );
        }

        // Add boost to non-zero segments
        const boosted = [baseAOverUnion, baseIntersectionOverUnion, baseBOverUnion].map((r) => r + BOOST);
        const total = boosted.reduce((sum, v) => sum + v, 0);
        const scale = total > 0 ? 100 / total : 0;

        return {
            aOnly: boosted[0] * scale,
            intersection: boosted[1] * scale,
            bOnly: boosted[2] * scale,
        };
    });

    // Final widths based on relativeTo
    const isUnion = $derived(relativeTo === "union");
    const pctOther = $derived(isUnion ? 100 : pctUnionPop);
    const pctBoth = $derived(isUnion ? unionSegments.aOnly + unionSegments.intersection : pctAWithIntersectionPop);
    const pctSelf = $derived(isUnion ? unionSegments.aOnly : pctAOnlyPop);
</script>

<div class="set-overlap-vis" title="A \ B: {countAOnly}, A∩B: {countIntersection} B \ A: {countBOnly}">
    <!-- Back to front: leftover (white) -> other/B-only -> both/intersection -> self/A-only -->
    <div class="bar leftover" class:hidden={isUnion}></div>
    <div class="bar" style="width: {pctOther}%; background: {rgbToCss(colors.setOverlap.other)}"></div>
    <div class="bar" style="width: {pctBoth}%; background: {rgbToCss(colors.setOverlap.both)}"></div>
    <div class="bar" style="width: {pctSelf}%; background: {rgbToCss(colors.setOverlap.self)}"></div>
</div>

<style>
    .set-overlap-vis {
        position: relative;
        width: 200px;
        height: 4px;
    }

    .bar {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        transition: width var(--transition-normal);
    }

    .leftover {
        width: 100%;
        background: white;
        transition: opacity var(--transition-normal);
    }

    .leftover.hidden {
        opacity: 0;
    }
</style>
