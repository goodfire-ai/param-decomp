import { backendGet } from "$lib/server/backend";
import type { ComponentDetail, SiteCurve } from "$lib/types";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ params }) => {
    const [detail, curve] = await Promise.all([
        backendGet<ComponentDetail>(
            `/api/runs/${params.run}/sites/${params.site}/components/${params.idx}`,
        ),
        backendGet<SiteCurve>(`/api/runs/${params.run}/sites/${params.site}/curve`),
    ]);
    return { run: params.run, site: params.site, detail, curve };
};
