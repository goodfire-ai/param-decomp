<script lang="ts">
    let { data }: { data: { site: string; listing: { total: number } } } = $props();
</script>

<svelte:head><title>Scope · {data.site}</title></svelte:head>

<div class="placeholder">
    <p class="eyebrow">{data.listing.total.toLocaleString("en-US")} components</p>
    <p class="subline">Select a component from the list to inspect it.</p>
</div>

<style>
    .placeholder {
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        text-align: center;
    }
</style>
