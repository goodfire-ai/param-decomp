<script lang="ts">
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import EdgeAttributionList from "./EdgeAttributionList.svelte";
    import SectionHeader from "./SectionHeader.svelte";

    type Props = {
        title: string;
        incomingLabel: string;
        outgoingLabel: string;
        incoming: EdgeAttribution[];
        outgoing: EdgeAttribution[];
        pageSize: number;
        onClick: (key: string) => void;
    };

    let { title, incomingLabel, outgoingLabel, incoming, outgoing, pageSize, onClick }: Props = $props();
</script>

{#if incoming.length > 0}
    <div class="edge-list-group">
        <SectionHeader title="{title} – {incomingLabel}" />
        <EdgeAttributionList items={incoming} {pageSize} {onClick} />
    </div>
{/if}

{#if outgoing.length > 0}
    <div class="edge-list-group">
        <SectionHeader title="{title} – {outgoingLabel}" />
        <EdgeAttributionList items={outgoing} {pageSize} {onClick} />
    </div>
{/if}

<style>
    .edge-list-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }
</style>
