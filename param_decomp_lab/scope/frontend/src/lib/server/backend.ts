import { env } from "$env/dynamic/private";
import { error } from "@sveltejs/kit";

const BACKEND_URL = env.SCOPE_BACKEND_URL ?? "http://localhost:8000";

/** Server-side fetch against the scope backend over localhost; 404s propagate as page 404s. */
export async function backendGet<T>(path: string): Promise<T> {
    const res = await fetch(`${BACKEND_URL}${path}`);
    if (res.status === 404) {
        const body = (await res.json()) as { detail: string };
        error(404, body.detail);
    }
    if (!res.ok) error(502, `scope backend returned ${res.status} for ${path}`);
    return (await res.json()) as T;
}
