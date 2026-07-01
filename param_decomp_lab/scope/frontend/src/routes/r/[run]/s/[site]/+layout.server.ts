import { backendGet } from "$lib/server/backend";
import type { Catalog, ComponentListing, SortKey } from "$lib/types";
import type { LayoutServerLoad } from "./$types";

const PAGE_SIZE = 100;
const SORT_KEYS: SortKey[] = ["mean_ci", "density", "max_act", "unlabeled_first"];

export const load: LayoutServerLoad = async ({ params, url }) => {
    const sortParam = url.searchParams.get("sort") ?? "mean_ci";
    const sort = SORT_KEYS.includes(sortParam as SortKey) ? (sortParam as SortKey) : "mean_ci";
    const page = Math.max(0, Number(url.searchParams.get("page") ?? "0") || 0);
    const q = url.searchParams.get("q") ?? "";

    const query = new URLSearchParams({
        sort,
        page: String(page),
        page_size: String(PAGE_SIZE),
        q,
    });
    const [catalog, listing] = await Promise.all([
        backendGet<Catalog>("/api/catalog"),
        backendGet<ComponentListing>(
            `/api/runs/${params.run}/sites/${params.site}/components?${query}`,
        ),
    ]);
    const run = catalog.runs.find((r) => r.run_id === params.run);
    const sites = run ? run.sites.filter((s) => s.n_components > 0).map((s) => s.site) : [];

    return {
        run: params.run,
        site: params.site,
        sites,
        sort,
        page,
        q,
        listing,
        pageSize: PAGE_SIZE,
    };
};
