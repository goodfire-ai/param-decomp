import { backendGet } from "$lib/server/backend";
import type { ComponentDetail } from "$lib/types";
import type { PageServerLoad } from "./$types";

const EXAMPLE_PAGE_SIZE = 20;

export const load: PageServerLoad = async ({ params, url }) => {
    const examplePage = Math.max(0, Number(url.searchParams.get("ep") ?? 0) | 0);
    const detail = await backendGet<ComponentDetail>(
        `/api/runs/${params.run}/sites/${params.site}/components/${params.idx}` +
            `?example_page=${examplePage}&example_page_size=${EXAMPLE_PAGE_SIZE}`,
    );
    return { run: params.run, site: params.site, detail, examplePageSize: EXAMPLE_PAGE_SIZE };
};
