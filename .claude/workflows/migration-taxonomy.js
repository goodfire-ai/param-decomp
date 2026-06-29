export const meta = {
  name: 'migration-taxonomy',
  description: 'Recursive feature taxonomy of `main` vs `feature/jax` — what the torch→JAX migration dropped / kept / changed',
  whenToUse: 'After the torch-shed work settles, before the feature/jax→main squash, to give the review a complete dropped/kept/changed map. Reads both branches at runtime.',
  phases: [
    { title: 'Discover', detail: 'map main into a tree of subsystems + modules' },
    { title: 'Taxonomize+Diff', detail: 'per module: recursive main taxonomy → 1:1 feature/jax mirror diff' },
    { title: 'Synthesize', detail: 'merge into one recursive taxonomy-diff doc' },
  ],
}

// Branch access: agents read `main` via `git show main:<path>` / `git ls-tree -r main`,
// and `feature/jax` from the working tree (or `git show feature/jax:<path>`). Both
// branches live in this checkout.

const SUBSYS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['subsystems'],
  properties: {
    subsystems: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'path', 'summary', 'modules'],
        properties: {
          name: { type: 'string' },
          path: { type: 'string', description: 'dir path on main, e.g. param_decomp_lab/harvest' },
          summary: { type: 'string' },
          modules: {
            type: 'array',
            description: 'the next-level functional units inside this subsystem (dirs/files)',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['name', 'path'],
              properties: { name: { type: 'string' }, path: { type: 'string' } },
            },
          },
        },
      },
    },
  },
}

const TAX_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['module', 'features'],
  properties: {
    module: { type: 'string' },
    features: {
      type: 'array',
      description: 'recursive feature list; hierarchy encoded in the dotted `path` (e.g. accumulator.reservoir_sampling)',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'what', 'key_files'],
        properties: {
          path: { type: 'string' },
          what: { type: 'string', description: 'the capability, not the code' },
          key_files: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
}

const DIFF_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['module', 'entries'],
  properties: {
    module: { type: 'string' },
    entries: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['feature_path', 'disposition', 'detail'],
        properties: {
          feature_path: { type: 'string', description: 'matches a main TAX feature path' },
          disposition: { type: 'string', enum: ['kept', 'changed', 'dropped', 'deferred', 'replaced'] },
          detail: { type: 'string', description: 'for changed/replaced: how; for dropped/deferred: why + where it went' },
          jax_location: { type: 'string', description: 'corresponding feature/jax path if kept/changed/replaced; empty otherwise' },
        },
      },
    },
  },
}

phase('Discover')
const discovery = await agent(
  `On the **main** branch (use \`git ls-tree -r main\` + \`git show main:<path>\`), survey the repo and map it into top-level functional SUBSYSTEMS (loosely by directory: e.g. param_decomp core, param_decomp/metrics, param_decomp_lab/{harvest,autointerp,clustering,app,experiments,eval_metrics,postprocess,infra}, param_decomp_config, param_decomp_jax, pretrain, ...). For each subsystem, list its next-level MODULES (the dirs/files that are distinct functional units). Output structured. Do NOT taxonomize features yet — just the subsystem→module skeleton.`,
  { schema: SUBSYS_SCHEMA, label: 'discover-main-tree' },
)

const modules = discovery.subsystems.flatMap((s) =>
  s.modules.map((m) => ({ subsystem: s.name, subsystem_path: s.path, name: m.name, path: m.path })),
)
log(`discovered ${discovery.subsystems.length} subsystems, ${modules.length} modules to taxonomize+diff`)

phase('Taxonomize+Diff')
const results = await pipeline(
  modules,
  // stage 1 — recursively taxonomize this module's FUNCTIONALITY on main
  (m) =>
    agent(
      `On the **main** branch, recursively taxonomize the FUNCTIONALITY of module \`${m.path}\` (subsystem ${m.subsystem}). Read it via \`git show main:<file>\` / \`git ls-tree -r main -- ${m.path}\`. Enumerate the capabilities it provides (recursively — encode the hierarchy in dotted feature paths), each with a one-line "what" and its key files. Be thorough; this is the "what we started with" record.`,
      { schema: TAX_SCHEMA, label: `tax:${m.subsystem}/${m.name}`, phase: 'Taxonomize+Diff' },
    ),
  // stage 2 — 1:1 mirror: find the corresponding feature/jax subtree, emit dropped/kept/changed
  (mainTax, m) =>
    agent(
      `Here is the main-branch feature taxonomy of \`${m.path}\`:\n${JSON.stringify(mainTax)}\n\n` +
        `Now search **feature/jax** (the working tree, or \`git show feature/jax:<path>\`) for the corresponding subtree (it may have moved, merged into param_decomp_jax, or been deleted). For EACH main feature above, emit a disposition: kept / changed / replaced / dropped / deferred, with detail (how it changed / where it went) and the feature/jax location if it survives. ` +
        `IMPORTANT: the torch→JAX run-adapter (loading old torch runs into JAX consumers) and "auto-decompose any eqx.Module" are **intentionally deferred** post-merge follow-ups — mark anything in that area \`deferred\`, NOT \`dropped\`. Ground every disposition in actual files/grep, not assumption.`,
      { schema: DIFF_SCHEMA, label: `diff:${m.subsystem}/${m.name}`, phase: 'Taxonomize+Diff' },
    ).then((diff) => ({ module: m, mainTax, diff })),
)

phase('Synthesize')
const clean = results.filter(Boolean)
const doc = await agent(
  `Merge these per-module (main taxonomy + feature/jax diff) results into ONE detailed, recursive **markdown** taxonomy doc and WRITE it to \`MIGRATION_TAXONOMY.md\` at the repo root (use the Write tool). ` +
    `Structure it as the feature tree we started with on main, grouped by subsystem → module → feature (hierarchy from the dotted paths), each annotated [kept|changed|replaced|dropped|deferred] with the detail + feature/jax location. ` +
    `Open with a summary table (counts per disposition + the headline narrative of what the migration dropped/kept/changed), and a short "deferred (post-merge)" section for the torch-bridge + auto-decompose. ` +
    `Flag any feature whose disposition is dropped/changed but which has NO note in MIGRATION_HOLES.md — those are the review's blind spots. ` +
    `Results:\n${JSON.stringify(clean)}`,
  { label: 'synthesize-taxonomy' },
)
return doc
