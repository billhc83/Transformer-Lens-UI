# TransformerLens Web UI

A production-quality web interface for mechanistic interpretability research on transformer models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![FastAPI](https://img.shields.io/badge/FastAPI-latest-orange.svg)](https://fastapi.tiangolo.com/) [![React 18](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/) [![CUDA](https://img.shields.io/badge/CUDA-recommended-green.svg)](https://developer.nvidia.com/cuda-downloads)

This app provides an interactive, dark-themed web interface that exposes all major capabilities of the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library. Researchers can visualize attention patterns, browse activation caches, run logit lens analyses, perform causal patching experiments, inspect circuits, write custom hooks, and stream live generation — all without writing Python scripts. All 10 planned phases are complete.

---

## Features

### Model Hub
- Browse 200+ HuggingFace transformer models with **LOCAL** or **DOWNLOAD** badge per model
- Load any model onto GPU with one click via `HookedTransformer.from_pretrained()`
- Architecture node graph: `embed → block_0 … block_N → ln_final → unembed` (React Flow)
- Model config summary displayed on load: `d_model`, `n_layers`, `n_heads`, `n_ctx`, `d_vocab`, `act_fn`

### Token Inspector
- Tokenize any text input; each token rendered as a coloured chip with position index
- Hover tooltip shows the token ID and token string
- Handles BOS tokens and subword tokenization correctly

### Forward Pass
- Run the full forward pass; see top-10 predicted next tokens for any position as a probability bar chart
- Click any token position to focus its prediction distribution
- Cyan gradient bars with probability percentages

### Activation Browser
- Runs `model.run_with_cache()` to capture all hook points (208 for GPT-2)
- Collapsible tree of hook point names grouped by block number, each showing tensor shape
- Tensor viewer: 1D bar chart, 2D heatmap with token-labeled rows, 3D with slice selector
- Stats chips display `min`, `max`, `mean`, `std` for any selected tensor

### Attention Visualizer
- Layer selector (slider + number input) to choose any transformer layer
- Thumbnail grid showing all attention heads as compact heatmaps
- Click any thumbnail to expand to full heatmap with query × key token labels on both axes
- Per-head stat chips and colour scale from black to cyan

### Logit Lens
- Applies final layer norm and unembed to the residual stream after each layer
- Shows top predicted tokens with probability bars for every layer row
- **✦ EMERGES** badge marks the layer where meaningful predictions first appear
- Validated on GPT-2: syntax tokens dominate L0–L8, location tokens emerge at L9

### Attribution Analyzer
- Decomposes logit attribution by component using `cache.decompose_resid()` + `cache.apply_ln_to_stack()`
- Three modes: **Full** (embed + pos_embed + per-layer attn/MLP), **By Layer**, **By Head**
- Plotly heatmap: components × positions, RdYlGn colorscale (red = negative, green = positive)
- Top contributors panel: position token selector + sorted bar chart of top 20 components
- Validated on GPT-2: L9, L11, L10 are top contributors for "Eiffel Tower → Paris"

### Patching Lab
- Causal activation patching to localize which layers/positions encode specific knowledge
- Split clean/corrupted prompt inputs; three patch types: **Residual**, **Attention Out**, **MLP Out**
- Uses `transformer_lens.patching` (`get_act_patch_resid_pre` / `get_act_patch_attn_out` / `get_act_patch_mlp_out`)
- Layered heatmap shows patching effect (logit diff change) per layer × position
- Verified on IOI task: `clean_diff=3.362`, `corrupted=-2.479`; mlp_out max at L0 pos10

### Circuit Analyzer
- React Flow node grid (layer × head) with composition score edges
- Edge colors: Q-composition = cyan, K-composition = purple, V-composition = green
- Threshold slider to filter low-score edges and reveal dominant circuits
- Right panel: QK and OV circuit SVD bar charts (top 10 singular values per selected head)
- Endpoints expose raw W_Q/W_K/W_V/W_O SVD decompositions for any head
- Verified on GPT-2: strongest K-composition is L1H8 → L3H7 (score 0.0255)

### Hook Lab
- Write and apply custom Python hook functions directly in the browser
- AST-validated sandboxed execution: no imports allowed; `torch` is injected into the namespace
- Four built-in templates: Zero Ablation, Mean Ablation, Scale × 0.5, Log
- Hook point filter to search the 208 GPT-2 hook names
- Active hooks panel with remove buttons; before/after top-5 prediction columns with change badge
- `POST /api/hooks/preview` runs a forward pass with and without active hooks for instant comparison
- Verified: zero-ablation on `blocks.5.attn.hook_z` produces a measurable shift in token probabilities

### Generation Studio
- Autoregressive streaming generation over WebSocket (`WS /ws/generate`)
- Token stream with cyan opacity encoding log-probability; click any token to inspect it
- Timeline scrubber to replay the generation step-by-step
- Right panel: token info, probability bar chart, live activation monitor stats
- Configure max tokens, temperature, and top-k; add hook monitor points before generation
- Verified on IOI prompt: first generated token is " Mary" (correct); attention monitor shape grows correctly

### AI Insights
- Available from every analysis page as an inline panel
- `POST /api/explain/insight` streams a 3–5 sentence mechanistic interpretability commentary via SSE
- Powered by a local Ollama `qwen3:14b` model — no data leaves your machine
- Commentary is specific to the current page data: names exact layers, heads, tokens, and values

---

## Prerequisites

- **Python 3.10 or higher**
- **Node.js 18 or higher + npm**
- **CUDA-capable GPU** recommended (CPU works but is slow for large models)
- **Git**
- **Ollama** with `qwen3:14b` pulled (required for AI Insights)

---

## Installation

1. Clone the repo:

    ```bash
    git clone <repo-url> && cd transformer_lens
    ```

2. Create Python virtual environment:

    ```bash
    python3 -m venv transformer-lens-env
    ```

3. Activate the virtual environment:

    ```bash
    source transformer-lens-env/bin/activate
    ```

4. Install Python dependencies:

    ```bash
    pip install transformer-lens fastapi uvicorn scipy
    ```

5. Install frontend dependencies:

    ```bash
    cd frontend && npm install && cd ..
    ```

> **Note:** The virtual environment is named `transformer-lens-env` (not `.venv`).  
> Models are downloaded from HuggingFace on first use and cached at `~/.cache/huggingface/hub`. Subsequent loads are instant.

---

## Running the App

### Backend

Run from the project root:

```bash
PYTHONPATH=. transformer-lens-env/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- Runs on port **8000**
- `PYTHONPATH=.` is required so Python finds the `backend` package
- Does **not** auto-reload; restart manually after code changes
- Swagger UI (interactive API docs) at [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend

Run from the `frontend/` directory:

```bash
cd frontend && npm run dev -- --host 0.0.0.0 --port 5173
```

- Runs on port **5173**
- Vite dev server proxies `/api/*` and `/ws/*` requests to the backend at port 8000
- Production build: `npm run build` (output in `frontend/dist/`)

---

## Usage

1. **Load a Model (Model Hub)**  
   Open [http://localhost:5173](http://localhost:5173). The Model Hub lists 200+ models. Models with a **LOCAL** badge are cached locally; **DOWNLOAD** will fetch from HuggingFace. Search for `gpt2`, click **Load**. The architecture graph renders once the model is ready.

2. **Tokenize text (Token Inspector)**  
   Navigate to **Token Inspector**. Type `The Eiffel Tower is in`, click **Tokenize**. 8 tokens appear as coloured chips (BOS + 7 words). Hover any chip to see its token ID.

3. **Run a forward pass (Forward Pass)**  
   Navigate to **Forward Pass**. Enter the same prompt and click **Run**. Click any token position to see the top-10 predicted next tokens as a bar chart. For the last position, London and Paris should appear in the top 10.

4. **Browse activations (Activation Browser)**  
   Click **Run Cache**. The tree on the left lists 208 hook keys for GPT-2. Expand a block group and click `blocks.5.attn.hook_pattern`. The right panel shows a 2D heatmap `[12, 8, 8]` with token-labeled rows and min/max/mean/std chips.

5. **Inspect attention heads (Attention)**  
   Set layer to **5**, click **Load Layer**. A grid of 12 thumbnail heatmaps appears. Click any head to expand to a full labelled heatmap with query and key token axes.

6. **Run logit lens (Logit Lens)**  
   The logit lens uses the most recent `run_with_cache` result automatically. Each row shows the model's top predictions after each layer. For GPT-2 on the Eiffel Tower prompt, the **✦ EMERGES** badge appears at layer 9.

7. **Decompose attribution (Attribution Analyzer)**  
   Enter an answer token (e.g., ` Paris`), select **By Layer**, and click **Run**. The heatmap shows which layers contribute most; the top contributors panel highlights L9, L11, and L10 for the Eiffel Tower prompt.

8. **Run a patching experiment (Patching Lab)**  
   Enter a clean prompt (`When Mary and John went to the store, John gave a drink to`) and a corrupted prompt (names swapped). Set answer token to ` Mary`, select **Residual**, and click **Run**. The heatmap reveals which layer/position stores the indirect object identity.

9. **Explore circuits (Circuit Analyzer)**  
   Click **Load Circuits**. The node grid renders all 144 heads for GPT-2. Drag the threshold slider down to reveal composition edges. Click any head node to inspect its QK and OV singular value decompositions in the right panel.

10. **Write a hook (Hook Lab)**  
    Select the **Zero Ablation** template, change the hook point to `blocks.5.attn.hook_z`, and click **Add Hook**. Click **Preview** to see how zeroing that head changes the top-5 predictions before and after.

11. **Stream generation (Generation Studio)**  
    Enter a prompt, adjust temperature and max tokens, optionally add monitor hook points, and click **Generate**. Tokens stream in one by one with cyan intensity encoding confidence. Click any token to inspect its probability distribution and activation stats.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check — returns `{"status": "ok"}` |
| GET | `/api/models/available` | List all 200+ models with `name` and `is_local` flag |
| POST | `/api/models/load` | Load a model by name onto GPU; returns config |
| GET | `/api/models/loaded` | Return currently loaded model info |
| POST | `/api/inference/tokenize` | Tokenize text; returns `token_ids`, `str_tokens`, `n_tokens` |
| POST | `/api/inference/forward` | Full forward pass; returns top-k predictions per token position |
| POST | `/api/inference/run_with_cache` | Run model with activation cache; stores all hook tensors server-side |
| POST | `/api/inference/logit_lens` | Logit lens across all layers; returns top-k predictions per layer |
| POST | `/api/inference/attribution` | Logit attribution by component; modes: `full`, `by_layer`, `by_head` |
| GET | `/api/activations/keys` | List all cached activation keys with their shapes |
| GET | `/api/activations/attention/{layer}` | All heads' attention patterns for a given layer |
| GET | `/api/activations/{key}` | Raw tensor data for a hook key (shape, stats, nested data array) |
| POST | `/api/patching/run` | Causal activation patching; returns logit diff heatmap per layer × position |
| GET | `/api/circuits/head_labels` | Head label list (`["L0H0"…]`), `n_layers`, `n_heads` |
| GET | `/api/circuits/qk/{layer}/{head}` | SVD of W_Q and W_K (top 10 singular values + vectors) |
| GET | `/api/circuits/ov/{layer}/{head}` | SVD of W_V and W_O (top 10 singular values + vectors) |
| GET | `/api/circuits/composition_scores` | Q/K/V composition scores for all head pairs |
| POST | `/api/hooks/add` | Register a sandboxed Python hook function |
| GET | `/api/hooks` | List all active hooks |
| DELETE | `/api/hooks/{id}` | Remove a hook by ID |
| POST | `/api/hooks/preview` | Forward pass with and without active hooks for comparison |
| POST | `/api/explain/insight` | SSE stream of AI Insights commentary for current page data |
| WS | `/ws/generate` | Stream token-by-token autoregressive generation with live activation stats |

---

## Architecture

The system has two processes that communicate over HTTP and WebSocket.

### FastAPI Backend (Python, port 8000)

- **ModelManager** singleton — loads models via `HookedTransformer.from_pretrained()`, maintains an LRU cache of up to 2 loaded models. Checks `~/.cache/huggingface/hub` before downloading.
- **CacheStore** singleton — holds the most recent `run_with_cache()` output as a `str → Tensor` dictionary, shared across requests.
- **HookStore** singleton — holds active hook functions registered via Hook Lab, applied via `model.run_with_hooks()`.
- Routers: `models.py`, `inference.py`, `activations.py`, `patching.py`, `circuits.py`, `hooks.py`, `explain.py`
- WebSocket handler in `main.py` for streaming generation
- Auto-generated OpenAPI docs at `/docs`

### React Frontend (TypeScript + Vite, port 5173)

- Zustand store tracks the currently loaded model configuration
- One page component per feature under `src/pages/`
- Reusable viz components: `AttentionHeatmap.tsx` (SVG heatmap with token axes), `ArchitectureGraph.tsx` (React Flow architecture graph), `CircuitGraph.tsx` (React Flow head grid with composition edges)
- Vite dev proxy forwards `/api/*` and `/ws/*` to `localhost:8000`

### Design Principles

- **Backend is the source of truth** — all model state, tensor computation, and activation storage live server-side
- **Stateless frontend** — the frontend holds only the current model name; all tensor data arrives as JSON
- **Local model priority** — the UI surfaces which models are cached locally vs. require a download
- **Local AI Insights** — commentary is generated by a local Ollama model; no data sent to external APIs

### Repository Structure

```
transformer_lens/
├── backend/
│   ├── main.py                   # FastAPI app entry point, CORS, router wiring, WS handler
│   ├── routers/
│   │   ├── models.py             # GET /api/models/available, POST /api/models/load
│   │   ├── inference.py          # tokenize, forward, run_with_cache, logit_lens, attribution
│   │   ├── activations.py        # activation keys, attention patterns, raw tensor fetch
│   │   ├── patching.py           # causal activation patching experiments
│   │   ├── circuits.py           # head labels, QK/OV SVD, composition scores
│   │   ├── hooks.py              # hook add/list/delete/preview with sandboxed exec
│   │   └── explain.py            # SSE AI Insights via local Ollama
│   └── services/
│       ├── model_manager.py      # Singleton LRU model loader
│       ├── cache_store.py        # Server-side activation cache
│       └── hook_store.py         # Active hook function registry
├── frontend/
│   └── src/
│       ├── App.tsx               # Sidebar navigation + page routing
│       ├── pages/                # One folder per feature page
│       │   ├── ModelHub/
│       │   ├── TokenInspector/
│       │   ├── ForwardPass/
│       │   ├── ActivationBrowser/
│       │   ├── AttentionViz/
│       │   ├── LogitLens/
│       │   ├── Attribution/
│       │   ├── PatchingLab/
│       │   ├── CircuitAnalyzer/
│       │   ├── HookLab/
│       │   └── GenerationStudio/
│       └── components/viz/
│           ├── AttentionHeatmap.tsx   # Reusable SVG heatmap
│           ├── ArchitectureGraph.tsx  # React Flow architecture graph
│           └── CircuitGraph.tsx       # React Flow head grid with composition edges
├── transformer-lens-env/         # Python virtual environment
└── PLAN.md                       # Full 10-phase implementation plan
```

---

## Design System

The UI uses a consistent AI-research aesthetic throughout:

| Token | Value | Usage |
|-------|-------|-------|
| Background | `#0a0a0f` | App background |
| Primary accent | `#00d4ff` (cyan) | Active states, selected items, highlights |
| Secondary accent | `#a855f7` (purple) | Node graph edges, phase badges |
| Danger | `#ff6b6b` (red) | Errors, negative attribution values |
| Font | JetBrains Mono | All text — tensors, labels, tokens |
| Panel style | Glassmorphism | `rgba(255,255,255,0.03)` + 1px border |

---

## Roadmap

- [x] Phase 1: Model Hub — model browser, GPU loading, architecture graph
- [x] Phase 2: Token Inspector + Forward Pass — tokenization, top-k predictions
- [x] Phase 3: Activation Browser — `run_with_cache`, hook point tree, tensor heatmaps
- [x] Phase 4: Attention Visualizer — per-head heatmap grid, expanded view
- [x] Phase 5: Logit Lens — layer-by-layer prediction buildup, EMERGES badge
- [x] Phase 6: Attribution Analyzer — logit attribution by head/MLP/layer with Plotly heatmap
- [x] Phase 7: Patching Lab — causal activation patching (clean vs. corrupted, IOI experiments)
- [x] Phase 8: Circuit Analyzer — QK/OV SVD, composition scores as weighted edges in React Flow
- [x] Phase 9: Hook Lab — sandboxed Python hook editor with before/after prediction comparison
- [x] Phase 10: Generation Studio — streaming token generation with live activation monitoring

---

## Contributing

The project follows a 10-phase plan described in [PLAN.md](PLAN.md). Each phase is self-contained and adds one or more pages.

1. Fork the repo and create a feature branch
2. Implement your changes following the existing structure
3. Verify against `gpt2` (local cache) before submitting
4. Open a Pull Request with a description of what phase/feature it covers

**Backend:** FastAPI + Python 3.10+. New endpoints go in the appropriate router; shared state goes in `services/`.

**Frontend:** React 18 + TypeScript (strict mode). One page per folder under `src/pages/`. Inline styles preferred over new CSS files; no CSS frameworks beyond TailwindCSS.

---

## License

MIT — see [LICENSE](LICENSE) for details.
