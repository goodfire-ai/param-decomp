import { backendGet } from "$lib/server/backend";
import type { ComponentListing, SiteCurve, SortKey } from "$lib/types";
import type { PageServerLoad } from "./$types";

const PAGE_SIZE = 50;
const SORT_KEYS: SortKey[] = ["density", "max_act", "unlabeled_first"];

export const load: PageServerLoad = async ({ params, url }) => {
    const sortParam = url.searchParams.get("sort") ?? "density";
    const sort = SORT_KEYS.includes(sortParam as SortKey) ? (sortParam as SortKey) : "density";
    const page = Math.max(0, Number(url.searchParams.get("page") ?? "0") || 0);
    const q = url.searchParams.get("q") ?? "";

    const query = new URLSearchParams({
        sort,
        page: String(page),
        page_size: String(PAGE_SIZE),
        q,
    });
    const [listing, curve] = await Promise.all([
        backendGet<ComponentListing>(
            `/api/runs/${params.run}/sites/${params.site}/components?${query}`,
        ),
        backendGet<SiteCurve>(`/api/runs/${params.run}/sites/${params.site}/curve`),
    ]);
    return { run: params.run, site: params.site, sort, page, q, listing, curve, pageSize: PAGE_SIZE };
};
