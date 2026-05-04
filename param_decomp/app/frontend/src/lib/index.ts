/**
 * A type that represents a value that may be uninitialized, loading, loaded, or in an error state.
 * This is useful for handling asynchronous data in a type-safe way.
 */
export type Loadable<T> =
    | { status: "uninitialized" }
    | { status: "loading" }
    | { status: "loaded"; data: T }
    | { status: "error"; error: unknown };

/** Map over the data inside a Loadable, preserving uninitialized/loading/error states */
export function mapLoadable<T, U>(loadable: Loadable<T>, fn: (data: T) => U): Loadable<U> {
    if (loadable.status === "uninitialized") return { status: "uninitialized" };
    if (loadable.status === "loading") return { status: "loading" };
    if (loadable.status === "error") return { status: "error", error: loadable.error };
    return { status: "loaded", data: fn(loadable.data) };
}
