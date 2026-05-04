# SPD Visualization App

A lightweight web app for visualizing SPD decomposition results. Built with Svelte 5 and FastAPI.

## Quick Start

**installation**

```bash
make install # install python project dependicies for backend
make install-app # install frontend dependencies
```

**Option 1: All-in-one launcher (recommended)**

```bash
make app
```

This automatically starts both backend and frontend with health checks and port detection.

**Option 2: Manual startup**

```bash
# in one terminal (backend)
uv run app/backend/server.py

# in another terminal (frontend)
cd app/frontend
npm run dev
```

## For ML Engineers/Researchers: Web Dev Basics

### JavaScript/Node.js Ecosystem

**package.json**: The Python equivalent of `pyproject.toml` or `requirements.txt`. Defines:

- Dependencies (like `numpy`, `torch` in Python)
- Scripts (like `make` commands)
- Metadata about the project

**npm** (Node Package Manager): Like `pip` or `uv` for Python packages.

**Common commands**:

```bash
npm install          # Install dependencies (like pip install -r requirements.txt)
npm run dev          # Start development server
npm run check        # Type check (like mypy or basedpyright)
npm run lint         # Check code for errors/style issues with ESLint (like ruff lint)
npm run format       # Auto-format code with Prettier (like ruff format)
```

### Svelte 5

**Key Svelte 5 features used in this app**:

- `$state(value)` - reactive state (replaces `let` variables)
- `$derived(expression)` - computed values (replaces `$:` statements)
- `$effect(() => {})` - side effects (replaces `onMount`, `afterUpdate`)
- `bind:value={variable}` - two-way binding
- `onclick={handler}` - event handlers (replaces `on:click`)
- `{#if condition}...{/if}` - conditional rendering

## Architecture

### Data Flow (End-to-End Example)

As an example, let's trace how loading a W&B run works:

1. **User Input** ([App.svelte](frontend/src/App.svelte))

   - User enters W&B run path in input field
   - Clicks "Load Run" button
   - `loadRun()` function is called
2. **Frontend API Call** ([api.ts:16-25](frontend/src/lib/api.ts))

   ```typescript
   export async function loadRun(wandbRunPath: string): Promise<void> {
     const url = new URL(`${API_URL}/runs/load`);
     // url-encode the wandb run path because it contains slashes
     const encodedWandbRunPath = encodeURIComponent(wandbRunPath);
     url.searchParams.set("wandb_run_path", encodedWandbRunPath);
     const response = await fetch(url.toString(), { method: "POST" });
     if (!response.ok) {
       const error = await response.json();
       throw new Error(error.detail || "Failed to load run");
     }
   }
   ```

3. **Backend Endpoint** ([server.py](backend/server.py))

   ```python
   @app.post("/runs/load")
   def load_run(wandb_run_path: str):
       run_context_service.load_run(unquote(wandb_run_path))
   ```

4. **Service Layer** ([run_context_service.py::RunContextService.load_run](backend/services/run_context_service.py))

   - Downloads ComponentModel from W&B
   - Loads model onto GPU
   - Creates data loader with tokenizer
   - Stores in `RunContextService` singleton


5. **UI Renders** ([App.svelte](frontend/src/App.svelte))
   - Sidebar shows config YAML
   - "Activation Contexts" tab becomes available

## Project Structure

```
app/
├── run_app.py                 # All-in-one launcher
├── run_backend.py             # Backend-only launcher
├── backend/
│   ├── server.py              # FastAPI routes
│   ├── schemas.py             # Pydantic models (API contracts)
│   ├── services/              # Business logic
│   └── lib/                   # Utilities
└── frontend/
    ├── package.json           # Dependencies & scripts
    ├── vite.config.ts         # Build tool config (minimal Vite setup)
    ├── svelte.config.js       # Svelte compiler config
    ├── index.html             # SPA entry point
    └── src/
        ├── main.ts            # TypeScript entry point
        ├── App.svelte         # Root component
        └── lib/
            ├── api.ts         # Backend API client
            └── index.ts       # Utility functions
        └── components/    # Svelte components
```

## Type Safety

Both frontend and backend use TypeScript/Python type annotations. However, the interface between them (API schemas) must be manually kept in sync:

- Backend: [schemas.py](backend/schemas.py) (Pydantic models)
- Frontend: [api.ts](frontend/src/lib/api.ts) (TypeScript types)

When you change the API, update both files.