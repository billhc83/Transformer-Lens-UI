from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json

router = APIRouter(prefix="/api/explain", tags=["explain"])

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:14b"

SYSTEM_PROMPT = (
    "You are a mechanistic interpretability expert analysing TransformerLens output. "
    "Give a specific, insightful interpretation of the data in 3–5 sentences. "
    "Skip generic definitions — go straight to what the numbers reveal about this model's behaviour. "
    "Be concrete: name specific layers, heads, tokens, or values from the data."
)


class ExplainRequest(BaseModel):
    page_type: str
    data: dict


# ── per-page prompt formatters ────────────────────────────────────────────────

def fmt_token_inspector(d: dict) -> str:
    tokens = list(zip(d.get("str_tokens", []), d.get("token_ids", [])))
    rows = "\n".join(f"  pos {i}: {repr(t)}  id={id_}" for i, (t, id_) in enumerate(tokens))
    return (
        f"GPT-2 tokenized {d.get('n_tokens')} tokens:\n{rows}\n\n"
        "What does this tokenization reveal? Note any multi-token words, surprising splits, "
        "or leading-space tokens that could affect the model's next-token prediction."
    )


def fmt_forward_pass(d: dict) -> str:
    preds = d.get("predictions", [])
    last = preds[-1] if preds else {}
    top = last.get("top_k", [])[:5]
    rows = "\n".join(
        f"  {e['token_str']!r:15s}  prob={e['probability']:.4f}  logit={e['logit']:.3f}"
        for e in top
    )
    return (
        f"Logits shape: {d.get('logits_shape')}\n"
        f"Tokens: {d.get('str_tokens')}\n"
        f"Final position predictions:\n{rows}\n\n"
        "What does this output distribution tell us about the model's confidence and "
        "which semantic category it has assigned to the prompt?"
    )


def fmt_activation_browser(d: dict) -> str:
    key = d.get("key", "unknown")
    shape = d.get("shape", [])
    stats = d.get("stats", {})
    return (
        f"Activation hook: {key}\n"
        f"Tensor shape: {shape}\n"
        f"Stats: mean={stats.get('mean', 'n/a'):.4f}, "
        f"max={stats.get('max', 'n/a'):.4f}, "
        f"min={stats.get('min', 'n/a'):.4f}, "
        f"std={stats.get('std', 'n/a'):.4f}\n\n"
        "What does this activation tell us about what this layer/module is computing? "
        "Consider the hook name, the shape (batch × seq × features), and the value range."
    )


def fmt_attention_viz(d: dict) -> str:
    layer = d.get("layer", "?")
    tokens = d.get("str_tokens", [])
    patterns = d.get("patterns", [])
    head_summaries = []
    for h, pat in enumerate(patterns[:12]):
        flat = [v for row in pat for v in row]
        if not flat:
            continue
        max_val = max(flat)
        max_idx = flat.index(max_val)
        seq = len(pat)
        row_i, col_j = divmod(max_idx, seq)
        t_from = tokens[row_i] if row_i < len(tokens) else f"pos{row_i}"
        t_to = tokens[col_j] if col_j < len(tokens) else f"pos{col_j}"
        head_summaries.append(f"  H{h}: strongest {repr(t_from)}→{repr(t_to)} ({max_val:.3f})")
    rows = "\n".join(head_summaries)
    return (
        f"Layer {layer} attention, tokens={tokens}\n"
        f"Strongest attention per head:\n{rows}\n\n"
        "Which heads show specialised behaviour (previous-token, duplicate-token, BOS-attending, "
        "subject-mover)? What does this suggest about the circuit active for this prompt?"
    )


def fmt_logit_lens(d: dict) -> str:
    tokens = d.get("str_tokens", [])
    results = d.get("results", [])
    target_pos = d.get("target_pos", len(tokens) - 1 if tokens else 0)
    emergence = d.get("emergence_layer")

    rows = []
    for lr in results:
        preds = lr.get("predictions", [])
        pos_pred = next((p for p in preds if p["position"] == target_pos), None)
        if pos_pred and pos_pred.get("top_k"):
            top1 = pos_pred["top_k"][0]
            rows.append(
                f"  L{lr['layer']:2d}: {top1['token_str']!r:12s}  {top1['probability']*100:.1f}%"
            )
    layer_table = "\n".join(rows)
    return (
        f"Logit Lens on tokens={tokens}, watching position {target_pos}:\n"
        f"{layer_table}\n"
        f"Emergence layer: {emergence}\n\n"
        "At which layer does the correct answer crystallise? Is this an attention or MLP contribution? "
        "What do the earlier predictions reveal about how the model processes the prompt?"
    )


def fmt_attribution(d: dict) -> str:
    mode = d.get("mode", "?")
    answer = d.get("answer_token_str", "?")
    tokens = d.get("str_tokens", [])
    top = d.get("top_contribs", [])[:10]
    rows = "\n".join(
        f"  {e['label']:12s}  score={e['score']:+.4f}"
        for e in top
    )
    return (
        f"Attribution ({mode} mode) for answer token {answer!r}\n"
        f"Prompt tokens: {tokens}\n"
        f"Top contributors:\n{rows}\n\n"
        "Which components drive the prediction most strongly? "
        "What do the negative scores mean? Does this align with known circuits for this task?"
    )


def fmt_patching_lab(d: dict) -> str:
    clean = d.get("clean_prompt", "")
    corrupt = d.get("corrupted_prompt", "")
    answer = d.get("answer_token", "")
    baseline = d.get("baseline_token", "")
    cdiff = d.get("clean_diff", 0)
    xdiff = d.get("corrupted_diff", 0)
    tokens = d.get("str_tokens", [])

    # Find top 5 patches across all activation types
    top_patches = []
    for act_type, matrix in d.get("results", {}).items():
        for layer_idx, row in enumerate(matrix):
            for pos_idx, val in enumerate(row):
                top_patches.append((val, act_type, layer_idx, pos_idx))
    top_patches.sort(key=lambda x: abs(x[0]), reverse=True)
    patch_rows = "\n".join(
        f"  {atype:12s} L{li:2d} pos{pi} ({tokens[pi] if pi < len(tokens) else '?'}): {v:+.3f}"
        for v, atype, li, pi in top_patches[:8]
    )
    return (
        f"Patching: clean={clean!r} → corrupted={corrupt!r}\n"
        f"Answer={answer!r}, baseline={baseline!r}\n"
        f"clean_diff={cdiff:.3f}, corrupted_diff={xdiff:.3f}\n"
        f"Tokens: {tokens}\n"
        f"Top causal sites:\n{patch_rows}\n\n"
        "Which layer/position combination causally stores the information that distinguishes "
        "clean from corrupted? Is it an MLP or attention effect? What does this imply about the circuit?"
    )


def fmt_circuit_analyzer(d: dict) -> str:
    labels = d.get("labels", [])
    n_layers = d.get("n_layers", "?")
    n_heads = d.get("n_heads", "?")

    edge_sections = []
    for comp_type in ("top_k_edges", "top_q_edges", "top_v_edges"):
        edges = d.get(comp_type, [])[:5]
        if edges:
            label = comp_type.replace("top_", "").replace("_edges", "").upper()
            rows = "\n".join(
                f"    {e['src']}→{e['dst']}: {e['score']:.4f}" for e in edges
            )
            edge_sections.append(f"  {label}-composition:\n{rows}")

    selected = d.get("selected_node")
    qk = d.get("qk_data")
    ov = d.get("ov_data")
    node_section = ""
    if selected:
        node_section = f"\nSelected head: {selected}\n"
        if qk:
            node_section += f"  QK singular values (top 5): {qk.get('S_Q', [])[:5]}\n"
        if ov:
            node_section += f"  OV singular values (top 5): {ov.get('S_V', [])[:5]}\n"

    return (
        f"Circuit Analyzer: {n_layers} layers × {n_heads} heads = {len(labels)} heads total\n"
        f"Top composition edges:\n" + "\n".join(edge_sections) + node_section + "\n\n"
        "Which compositional relationships form a coherent circuit? "
        "Does the K-composition pattern suggest induction heads? "
        "What does the OV spectrum tell us about the head's function?"
    )


def fmt_hook_lab(d: dict) -> str:
    hook = d.get("hook_name", "?")
    code = d.get("hook_code", "").strip()
    baseline = d.get("baseline", [])[:5]
    modified = d.get("modified", [])[:5]
    changed = d.get("changed", False)

    base_rows = "\n".join(f"  {e['token']!r:15s} {e['prob']*100:.2f}%" for e in baseline)
    mod_rows = "\n".join(f"  {e['token']!r:15s} {e['prob']*100:.2f}%" for e in modified)
    return (
        f"Hook: {hook}\nCode:\n{code}\n\n"
        f"Baseline top predictions:\n{base_rows}\n\n"
        f"Modified top predictions:\n{mod_rows}\n\n"
        f"Top prediction changed: {changed}\n\n"
        "What does this intervention tell us about the causal role of this hook point? "
        "If the prediction changed, what information was removed? "
        "If unchanged, does that mean this hook isn't part of the circuit, or is there redundancy?"
    )


def fmt_sae_studio(d: dict) -> str:
    token_activation_rows = "\n".join([f"Token: {token}, Activation: {activation}" for token, activation in zip(d['str_tokens'], d['activations'])])

    topTokens = sorted(zip(d['str_tokens'], d['activations']), key=lambda x: x[1], reverse=True)[:3]
    top3_tokens = "\n".join([f"  - {token} (Activation: {activation})" for token, activation in topTokens])

    if d['layer'] <= 3:
        layer_stage = "early (syntax/position)"
    elif d['layer'] <= 7:
        layer_stage = "mid (semantics)"
    else:
        layer_stage = "late (facts/entities)"

    return f"""
For Layer {d['layer']} ({layer_stage}) in Feature ID: {d['feature_id']},
Detected Tokens and Activations:
{token_activation_rows}

Top 3 Active Tokens:
{top3_tokens}

Neuronpedia URL: {d['neuronpedia_url']}

What specific linguistic or semantic concept does this feature detect?
"""


def fmt_generation_studio(d: dict) -> str:
    tokens = d.get("tokens", [])
    prompt = d.get("prompt", "")
    temperature = d.get("temperature", 1.0)
    generated = "".join(t["token"] for t in tokens)
    logprobs = [(t["token"], t["logprob"]) for t in tokens]
    low_conf = [(tok, lp) for tok, lp in logprobs if lp < -2.5]
    lp_rows = "\n".join(
        f"  step {t['step']:3d}: {t['token']!r:15s} logprob={t['logprob']:.3f}"
        for t in tokens[:20]
    )
    return (
        f"Generation Studio\nPrompt: {prompt!r}\nTemperature: {temperature}\n"
        f"Generated: {generated!r}\n\n"
        f"Token logprobs (first 20 steps):\n{lp_rows}\n"
        f"Low-confidence tokens (logprob < -2.5): {low_conf}\n\n"
        "Where does the model show high vs low confidence during generation? "
        "Are there specific tokens where uncertainty spikes? "
        "What does this suggest about the model's internal representation at those steps?"
    )


def fmt_normalization_probe(data: dict) -> str:
    concept = data.get("concept", "unknown")
    sway = data.get("sway_score", 0)
    mean_kl = data.get("mean_kl", 0)
    probe_count = data.get("probe_count", 0)
    top_tokens = data.get("top_consistent_tokens", [])
    token_rows = "\n".join(
        f"  {t['token']!r}: mean Δ={t['mean_delta']:+.4f}, consistency={t['consistency_pct']:.0f}%"
        for t in top_tokens[:10]
    )
    return (
        f"Normalization Probe\nConcept: {concept!r}\n"
        f"Sway score: {sway:.4f}  Mean KL: {mean_kl:.4f}  Probes: {probe_count}\n\n"
        f"Most consistently shifted tokens:\n{token_rows or '  (none above threshold)'}\n\n"
        "What do these results tell us about how normalization-framing cues affect the model's "
        "next-token predictions? Are the consistently shifted tokens semantically related to the concept? "
        "Does the sway score suggest a meaningful effect or noise? "
        "What mechanistic interpretability follow-up (logit lens, attribution, patching) would best "
        "isolate the layer or head responsible for this shift?"
    )


FORMATTERS = {
    "token-inspector": fmt_token_inspector,
    "forward-pass": fmt_forward_pass,
    "activation-browser": fmt_activation_browser,
    "attention-viz": fmt_attention_viz,
    "logit-lens": fmt_logit_lens,
    "attribution": fmt_attribution,
    "patching-lab": fmt_patching_lab,
    "circuit-analyzer": fmt_circuit_analyzer,
    "hook-lab": fmt_hook_lab,
    "generation-studio": fmt_generation_studio,
    "sae-studio": fmt_sae_studio,
    "normalization-probe": fmt_normalization_probe,
}


async def stream_ollama(prompt: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": False,
        "options": {"temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


@router.post("")
async def explain(req: ExplainRequest):
    formatter = FORMATTERS.get(req.page_type)
    if formatter:
        prompt = formatter(req.data)
    else:
        prompt = f"Explain this TransformerLens data:\n{json.dumps(req.data, indent=2)[:3000]}"
    return StreamingResponse(stream_ollama(prompt), media_type="text/plain; charset=utf-8")
