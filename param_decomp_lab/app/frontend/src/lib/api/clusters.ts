/**
 * API client for /api/clusters endpoints.
 */

import { apiUrl } from "./index";

export type ClusterMapping = {
    mapping: Record<string, number>;
};

export async function loadClusterMapping(filePath: string): Promise<ClusterMapping> {
    const url = apiUrl("/api/clusters/load");
    url.searchParams.set("file_path", filePath);

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to load cluster mapping");
    }

    return (await response.json()) as ClusterMapping;
}
