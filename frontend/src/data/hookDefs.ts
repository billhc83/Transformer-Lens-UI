export const HOOK_DEFS: Record<string, string> = {
  "hook_embed": "Embeds input tokens into continuous vector representations",
  "hook_pos_embed": "Adds positional information to token embeddings",
  "hook_tokens": "Captures input token representations before processing",
  "blocks.{N}.hook_resid_pre": "Residual stream before block {N}",
  "blocks.{N}.hook_resid_mid": "Residual stream after attention in block {N}",
  "blocks.{N}.hook_resid_post": "Residual stream after full block {N}",
  "blocks.{N}.attn.hook_q": "Query vectors from attention head in block {N}",
  "blocks.{N}.attn.hook_k": "Key vectors from attention head in block {N}",
  "blocks.{N}.attn.hook_v": "Value vectors from attention head in block {N}",
  "blocks.{N}.attn.hook_z": "Attention layer output before applying value vectors",
  "blocks.{N}.attn.hook_attn_scores": "Attention scores between query and key vectors",
  "blocks.{N}.attn.hook_pattern": "Attention distribution matrix between tokens",
  "blocks.{N}.hook_attn_out": "Output of attention mechanism before MLP",
  "blocks.{N}.hook_mlp_out": "Output of MLP after nonlinear transformations",
  "blocks.{N}.mlp.hook_pre": "Input to MLP before first linear transformation",
  "blocks.{N}.mlp.hook_post": "Output of MLP after final nonlinear transformation",
  "ln_final.hook_scale": "Scaling factor from final layer normalization",
  "ln_final.hook_normalized": "Input to final linear layer after normalization",
  "hook_result": "Combined output from all attention heads",
  "blocks.{N}.attn.hook_result": "Output of single attention head {N}",
  "blocks.{N}.ln1.hook_scale": "Scaling factor from first block normalization",
  "blocks.{N}.ln1.hook_normalized": "Normalized input for the attention in block {N}",
  "blocks.{N}.ln2.hook_scale": "Scaling factor from second block normalization",
  "blocks.{N}.ln2.hook_normalized": "Normalized result before MLP in block {N}",
  "blocks.{N}.attn.hook_rot_q": "Rotated query vectors from attention head",
  "blocks.{N}.attn.hook_rot_k": "Rotated key vectors from attention head",
  "hook_scale": "Scaling factor from standalone layer normalization",
  "blocks.{N}.mlp.hook_pre_linear": "Input before first linear transformation in MLP",
  "blocks.{N}.attn.W_Q": "Weight matrix for query vectors in block {N}",
  "blocks.{N}.attn.W_K": "Weight matrix for key vectors in block {N}",
  "blocks.{N}.attn.W_V": "Weight matrix for value vectors in block {N}",
  "blocks.{N}.attn.W_O": "Output weight matrix in block {N}",
}

export function resolveHookDef(key: string): string | null {
  const normalized = key.replace(/blocks\.\d+\./g, 'blocks.{N}.')
  return HOOK_DEFS[normalized] ?? HOOK_DEFS[key] ?? null
}
