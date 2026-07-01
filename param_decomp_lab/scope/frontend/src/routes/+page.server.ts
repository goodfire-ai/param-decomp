import { backendGet } from "$lib/server/backend";
import type { Catalog } from "$lib/types";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async () => {
    const catalog = await backendGet<Catalog>("/api/catalog");
    return { catalog };
};
