import { backendGet } from "$lib/server/backend";
import type { ComponentDetail } from "$lib/types";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ params }) => {
    const detail = await backendGet<ComponentDetail>(
        `/api/runs/${params.run}/sites/${params.site}/components/${params.idx}`,
    );
    return { run: params.run, site: params.site, detail };
};
